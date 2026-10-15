/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Round-trip tests for copyHostToDevice / copyDeviceToHost in kvCacheManagerV2Utils.cu.
//
// A call whose tasks are at least hostCopyDmaMinBytes() each takes the copy-engine path (one cuMemcpyBatchAsync, or
// one cuMemcpy{HtoD,DtoH}Async per task on the NULL stream, which the batched call does not accept); smaller tasks take
// the batched copy kernel. The knob is read once per process, so instead of changing the environment variable
// mid-process the tests pick sizes on both sides of the default and reach both branches that way.

#include "tensorrt_llm/batch_manager/kvCacheManagerV2Utils.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/batchedPageCopy.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMem.h"
#include "tensorrt_llm/common/tllmException.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

using MMTask = Task<MemAddress, MemAddress>;

// Page sizes of a production hybrid-attention recipe: an attention page, which stays on the kernel at the default
// threshold, and the conv-state and bf16 SSM-state pages of its recurrent layers, which take the copy engines.
constexpr size_t kAttentionPageBytes = 884736;
constexpr size_t kConvStateBytes = 15261696;
constexpr size_t kSsmStateBytes = 217055232;
constexpr size_t kDefaultThreshold = size_t{4} << 20; // kvCacheManagerV2Utils.cu default

struct CudaDeleter
{
    void operator()(void* ptr) const noexcept
    {
        cudaFree(ptr);
    }
};

using CudaAllocation = std::unique_ptr<void, CudaDeleter>;

CudaAllocation allocateCuda(size_t numBytes)
{
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, numBytes) != cudaSuccess)
    {
        return CudaAllocation{};
    }
    return CudaAllocation{ptr};
}

class ScopedEnv
{
public:
    ScopedEnv(char const* name, std::string const& value)
        : mName(name)
    {
        if (char const* oldValue = std::getenv(name); oldValue != nullptr)
        {
            mOldValue = oldValue;
            mHadOldValue = true;
        }
        if (::setenv(name, value.c_str(), /*overwrite=*/1) != 0)
        {
            throw std::system_error(errno, std::generic_category(), "setenv failed");
        }
    }

    ~ScopedEnv()
    {
        if (mHadOldValue)
        {
            ::setenv(mName.c_str(), mOldValue.c_str(), /*overwrite=*/1);
        }
        else
        {
            ::unsetenv(mName.c_str());
        }
    }

private:
    std::string mName;
    std::string mOldValue;
    bool mHadOldValue = false;
};

// The integer value of an environment variable, or nullopt when it is unset or not an integer, which is how the
// policy reads its two knobs.
std::optional<long long> integerEnv(char const* name)
{
    char const* const env = std::getenv(name);
    if (env == nullptr || *env == '\0')
    {
        return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    long long const value = std::strtoll(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0')
    {
        return std::nullopt;
    }
    return value;
}

// What hostCopyDmaMinBytes() must return for the environment this process started with.
size_t expectedThresholdFromEnv()
{
    auto const parsed = integerEnv("TLLM_KV_CACHE_MANAGER_V2_HOST_COPY_DMA_MIN_BYTES");
    if (!parsed.has_value())
    {
        return kDefaultThreshold;
    }
    return *parsed < 0 ? std::numeric_limits<size_t>::max() : static_cast<size_t>(*parsed);
}

void fillPattern(unsigned char* bytes, size_t numBytes, unsigned char seed)
{
    for (size_t i = 0; i < numBytes; ++i)
    {
        bytes[i] = static_cast<unsigned char>((i * 131U + seed) & 0xFFU);
    }
}

size_t roundUpToHostMemAlignment(size_t numBytes)
{
    return (numBytes + HostMem::kAlignment - 1) / HostMem::kAlignment * HostMem::kAlignment;
}

// HostMem (pinned, cuMemHostRegister'ed like the production host pools) -> device -> HostMem, byte-exact.
// numTasks contiguous slots of numBytes each, the slot layout of a host slot pool (base + slot * slot_size).
void roundTrip(size_t numBytes, size_t numTasks, cudaStream_t stream)
{
    // Keep HostMem construction cheap in the test: no prefault threads, regular pages.
    ScopedEnv prefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "0");
    ScopedEnv thp("TLLM_KV_CACHE_MANAGER_V2_THP", "0");

    size_t const total = numBytes * numTasks;
    HostMem src(roundUpToHostMemAlignment(total));
    HostMem dst(roundUpToHostMemAlignment(total));
    CudaAllocation device = allocateCuda(total);
    ASSERT_NE(device.get(), nullptr);

    // Drain the stream before the buffers above go away, also when an assertion below returns early.
    struct DrainOnExit
    {
        cudaStream_t stream;

        ~DrainOnExit()
        {
            cudaStreamSynchronize(stream);
        }
    } const drain{stream};

    auto* srcBytes = reinterpret_cast<unsigned char*>(src.address());
    auto* dstBytes = reinterpret_cast<unsigned char*>(dst.address());
    fillPattern(srcBytes, total, static_cast<unsigned char>(numTasks));
    std::memset(dstBytes, 0, total);
    ASSERT_EQ(cudaMemsetAsync(device.get(), 0xEE, total, stream), cudaSuccess);

    auto const deviceBase = reinterpret_cast<MemAddress>(device.get());
    std::vector<MMTask> hostToDevice;
    std::vector<MMTask> deviceToHost;
    hostToDevice.reserve(numTasks);
    deviceToHost.reserve(numTasks);
    for (size_t i = 0; i < numTasks; ++i)
    {
        hostToDevice.push_back(MMTask{deviceBase + i * numBytes, src.address() + i * numBytes});
        deviceToHost.push_back(MMTask{dst.address() + i * numBytes, deviceBase + i * numBytes});
    }

    ASSERT_EQ(copyHostToDevice(hostToDevice, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(stream)),
        CUDA_SUCCESS);
    ASSERT_EQ(copyDeviceToHost(deviceToHost, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(stream)),
        CUDA_SUCCESS);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(std::memcmp(srcBytes, dstBytes, total), 0) << "numBytes=" << numBytes << " numTasks=" << numTasks;
}

class KvCacheManagerV2HostCopyTest : public testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking), cudaSuccess);
    }

    void TearDown() override
    {
        if (mStream != nullptr)
        {
            cudaStreamDestroy(mStream);
        }
    }

    cudaStream_t mStream = nullptr;
};

TEST_F(KvCacheManagerV2HostCopyTest, ThresholdFollowsEnvironment)
{
    EXPECT_EQ(hostCopyDmaMinBytes(), expectedThresholdFromEnv());
    bool const chunked = HostMem::shouldUseChunkedRegistration();
    for (size_t numBytes :
        {size_t{16}, kAttentionPageBytes, kDefaultThreshold - 1, kDefaultThreshold, kConvStateBytes, kSsmStateBytes})
    {
        bool const expected = !chunked && numBytes >= hostCopyDmaMinBytes();
        EXPECT_EQ(useCopyEngineForHostCopy(numBytes), expected) << "numBytes=" << numBytes;
    }
}

// Both branches accept an empty call and enqueue nothing.
TEST_F(KvCacheManagerV2HostCopyTest, EmptyTaskListIsANoOp)
{
    std::vector<MMTask> const none;
    for (size_t numBytes : {kAttentionPageBytes, kSsmStateBytes})
    {
        EXPECT_EQ(
            copyHostToDevice(none, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(mStream)), CUDA_SUCCESS);
        EXPECT_EQ(
            copyDeviceToHost(none, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(mStream)), CUDA_SUCCESS);
    }
    EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
}

class HostCopyRoundTripTest : public KvCacheManagerV2HostCopyTest,
                              public testing::WithParamInterface<std::pair<size_t, size_t>>
{
};

TEST_P(HostCopyRoundTripTest, HostToDeviceToHost)
{
    auto const [numBytes, numTasks] = GetParam();
    roundTrip(numBytes, numTasks, mStream);
}

// 300 tasks > 256 covers the kernel path's multi-launch remainder; the state pages take the copy-engine path at
// the default threshold. (kDefaultThreshold, 8), (kDefaultThreshold, 9) and (kConvStateBytes, 12) are batches of
// several large tasks that travel in one cuMemcpyBatchAsync.
INSTANTIATE_TEST_SUITE_P(Shapes, HostCopyRoundTripTest,
    testing::Values(std::make_pair(kAttentionPageBytes, size_t{1}), std::make_pair(kAttentionPageBytes, size_t{140}),
        std::make_pair(kAttentionPageBytes, size_t{300}), std::make_pair(size_t{16}, size_t{300}),
        std::make_pair(kConvStateBytes, size_t{1}), std::make_pair(kSsmStateBytes, size_t{1}),
        std::make_pair(kDefaultThreshold, size_t{8}), std::make_pair(kDefaultThreshold, size_t{9}),
        std::make_pair(kConvStateBytes, size_t{12})));

// The kernel requires numBytes % 16 == 0 and rejects anything else with TLLM_CHECK before enqueuing work; the
// copy-engine path has no such constraint. Which path a size takes depends on the threshold this process started
// with (24 B is below the default, 4 MiB + 8 B above it), so each size is either round-tripped or asserted to throw
// according to the policy predicate.
TEST_F(KvCacheManagerV2HostCopyTest, NonMultipleOf16)
{
    size_t const numTasks = 3;
    for (size_t numBytes : {size_t{24}, kDefaultThreshold + 8})
    {
        if (useCopyEngineForHostCopy(numBytes))
        {
            roundTrip(numBytes, numTasks, mStream);
            continue;
        }
        // Rejected before any CUDA work: the addresses are never dereferenced, so they need not be real.
        std::vector<MMTask> const tasks(numTasks, MMTask{MemAddress{0x1000}, MemAddress{0x2000}});
        EXPECT_THROW(copyHostToDevice(tasks, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(mStream)),
            tensorrt_llm::common::TllmException)
            << "numBytes=" << numBytes;
        EXPECT_THROW(copyDeviceToHost(tasks, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(mStream)),
            tensorrt_llm::common::TllmException)
            << "numBytes=" << numBytes;
    }
}

// The kernel launch is legal on the NULL stream; the copy-engine path falls back to one cuMemcpy{HtoD,DtoH}Async per
// task there, because cuMemcpyBatchAsync does not accept the legacy NULL stream. Both must round-trip.
TEST_F(KvCacheManagerV2HostCopyTest, NullStreamAccepted)
{
    roundTrip(kAttentionPageBytes, 4, nullptr);
    roundTrip(kConvStateBytes, 1, nullptr);
    roundTrip(kConvStateBytes, 3, nullptr);
}

// Work queued earlier on the stream must be visible to the copy, and a second stream waiting on an event recorded
// after the copy must see the copied bytes (the runtime tracks completion only through such events).
TEST_F(KvCacheManagerV2HostCopyTest, StreamOrderingThroughEvents)
{
    ScopedEnv prefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "0");
    ScopedEnv thp("TLLM_KV_CACHE_MANAGER_V2_THP", "0");

    for (size_t numBytes : {kAttentionPageBytes, kConvStateBytes})
    {
        HostMem host(roundUpToHostMemAlignment(numBytes));
        CudaAllocation device = allocateCuda(numBytes);
        CudaAllocation deviceCheck = allocateCuda(numBytes);
        ASSERT_NE(device.get(), nullptr);
        ASSERT_NE(deviceCheck.get(), nullptr);
        auto* hostBytes = reinterpret_cast<unsigned char*>(host.address());
        std::memset(hostBytes, 0, numBytes);

        // Producer on mStream: fill device, then device -> host, then record.
        ASSERT_EQ(cudaMemsetAsync(device.get(), 0xA5, numBytes, mStream), cudaSuccess);
        std::vector<MMTask> deviceToHost{MMTask{host.address(), reinterpret_cast<MemAddress>(device.get())}};
        ASSERT_EQ(copyDeviceToHost(deviceToHost, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(mStream)),
            CUDA_SUCCESS);
        cudaEvent_t ready = nullptr;
        ASSERT_EQ(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(ready, mStream), cudaSuccess);

        // Consumer on a second stream: wait on the event, host -> deviceCheck, read back.
        cudaStream_t consumer = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaStreamWaitEvent(consumer, ready, 0), cudaSuccess);
        std::vector<MMTask> hostToDevice{MMTask{reinterpret_cast<MemAddress>(deviceCheck.get()), host.address()}};
        ASSERT_EQ(copyHostToDevice(hostToDevice, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(consumer)),
            CUDA_SUCCESS);
        std::vector<unsigned char> readBack(numBytes);
        ASSERT_EQ(cudaMemcpyAsync(readBack.data(), deviceCheck.get(), numBytes, cudaMemcpyDeviceToHost, consumer),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
        EXPECT_TRUE(std::all_of(readBack.begin(), readBack.end(), [](unsigned char v) { return v == 0xA5; }))
            << "numBytes=" << numBytes;
        EXPECT_TRUE(std::all_of(hostBytes, hostBytes + numBytes, [](unsigned char v) { return v == 0xA5; }))
            << "numBytes=" << numBytes;

        ASSERT_EQ(cudaStreamDestroy(consumer), cudaSuccess);
        ASSERT_EQ(cudaEventDestroy(ready), cudaSuccess);
    }
}

// Informational only: event-to-event GB/s of one SSM-state page in each direction and the path that moved it.
// Not asserted, because the figure depends on the link, the host NUMA placement and whatever else shares the
// machine. On the system that motivated the change the kernel path moved such a page at 1.8-17.5 GB/s per task and
// the copy engines in well under a millisecond.
TEST_F(KvCacheManagerV2HostCopyTest, StateSlotBandwidthInfo)
{
    ScopedEnv prefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "0");
    ScopedEnv thp("TLLM_KV_CACHE_MANAGER_V2_THP", "1");

    size_t const numBytes = kSsmStateBytes;
    HostMem host(roundUpToHostMemAlignment(numBytes));
    CudaAllocation device = allocateCuda(numBytes);
    ASSERT_NE(device.get(), nullptr);
    std::memset(reinterpret_cast<void*>(host.address()), 0x3C, numBytes);
    std::vector<MMTask> hostToDevice{MMTask{reinterpret_cast<MemAddress>(device.get()), host.address()}};
    std::vector<MMTask> deviceToHost{MMTask{host.address(), reinterpret_cast<MemAddress>(device.get())}};

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

    auto timeGbPerSec = [&](auto&& issue) -> double
    {
        constexpr int kWarmup = 3;
        constexpr int kIters = 10;
        for (int i = 0; i < kWarmup; ++i)
        {
            issue();
        }
        EXPECT_EQ(cudaEventRecord(start, mStream), cudaSuccess);
        for (int i = 0; i < kIters; ++i)
        {
            issue();
        }
        EXPECT_EQ(cudaEventRecord(stop, mStream), cudaSuccess);
        EXPECT_EQ(cudaEventSynchronize(stop), cudaSuccess);
        float ms = 0.0F;
        EXPECT_EQ(cudaEventElapsedTime(&ms, start, stop), cudaSuccess);
        return static_cast<double>(numBytes) * kIters / (static_cast<double>(ms) * 1e-3) / 1e9;
    };

    double const h2d = timeGbPerSec(
        [&]
        {
            EXPECT_EQ(
                copyHostToDevice(hostToDevice, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(mStream)),
                CUDA_SUCCESS);
        });
    double const d2h = timeGbPerSec(
        [&]
        {
            EXPECT_EQ(
                copyDeviceToHost(deviceToHost, static_cast<ssize_t>(numBytes), reinterpret_cast<CUstream>(mStream)),
                CUDA_SUCCESS);
        });
    bool const copyEngine = useCopyEngineForHostCopy(numBytes);
    std::cout << "StateSlotBandwidthInfo: " << numBytes << " B page, path=" << (copyEngine ? "copy-engine" : "kernel")
              << ", host->device " << h2d << " GB/s, device->host " << d2h << " GB/s" << std::endl;
    RecordProperty("path", copyEngine ? "copy-engine" : "kernel");
    RecordProperty("host_to_device_gb_per_s", std::to_string(h2d));
    RecordProperty("device_to_host_gb_per_s", std::to_string(d2h));

    ASSERT_EQ(cudaEventDestroy(start), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(stop), cudaSuccess);
}

// TEMPORARY BENCHMARK (not for the PR): event-to-event GB/s of the two entry points of this file against
// BatchedPageCopier, the path StorageManager uses for hot<->host migrations on main, on the same shapes.
TEST_F(KvCacheManagerV2HostCopyTest, PathComparisonInfo)
{
    ScopedEnv prefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "0");
    ScopedEnv thp("TLLM_KV_CACHE_MANAGER_V2_THP", "1");

    BatchedPageCopier const copier;
    auto const& topo = copier.topology();
    std::cout << "PathComparisonInfo: coherentLink=" << topo.coherentLink << " linkGBs=" << topo.linkBandwidthGBs
              << " smCount=" << topo.smCount << " hostNuma=" << topo.hostNumaId
              << " kernelEnabled=" << copier.kernelEnabled()
              << " ctas(d2h/h2d)=" << copier.config(CopyDirection::kD2H).ctas << "/"
              << copier.config(CopyDirection::kH2D).ctas
              << " inFlight(d2h/h2d)=" << copier.config(CopyDirection::kD2H).inFlightBytes << "/"
              << copier.config(CopyDirection::kH2D).inFlightBytes << std::endl;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

    struct Shape
    {
        size_t numBytes;
        size_t numTasks;
        char const* name;
    };

    for (Shape const shape : {Shape{kSsmStateBytes, 1, "ssm-state x1"}, Shape{kConvStateBytes, 12, "conv-state x12"},
             Shape{kConvStateBytes, 1, "conv-state x1"}, Shape{kAttentionPageBytes, 140, "attention x140"},
             Shape{kDefaultThreshold, 64, "4MiB x64"}})
    {
        size_t const total = shape.numBytes * shape.numTasks;
        HostMem host(roundUpToHostMemAlignment(total));
        CudaAllocation device = allocateCuda(total);
        ASSERT_NE(device.get(), nullptr);
        std::memset(reinterpret_cast<void*>(host.address()), 0x3C, total);
        auto const deviceBase = reinterpret_cast<MemAddress>(device.get());

        std::vector<MMTask> hostToDevice;
        std::vector<MMTask> deviceToHost;
        std::vector<PageIndexPair> pairs;
        for (size_t i = 0; i < shape.numTasks; ++i)
        {
            hostToDevice.push_back(MMTask{deviceBase + i * shape.numBytes, host.address() + i * shape.numBytes});
            deviceToHost.push_back(MMTask{host.address() + i * shape.numBytes, deviceBase + i * shape.numBytes});
            pairs.push_back(PageIndexPair{static_cast<int32_t>(i), static_cast<int32_t>(i)});
        }
        CudaAllocation devicePairs;
        PageIndexPair const* pairPtr = pairs.data();
        if (copier.pageIndexLocation() == PageIndexLocation::kDevice)
        {
            devicePairs = allocateCuda(pairs.size() * sizeof(PageIndexPair));
            ASSERT_NE(devicePairs.get(), nullptr);
            ASSERT_EQ(cudaMemcpy(devicePairs.get(), pairs.data(), pairs.size() * sizeof(PageIndexPair),
                          cudaMemcpyHostToDevice),
                cudaSuccess);
            pairPtr = static_cast<PageIndexPair const*>(devicePairs.get());
        }
        PoolCopyArgs onboard{};
        onboard.dstBase = static_cast<CUdeviceptr>(deviceBase);
        onboard.srcBase = static_cast<CUdeviceptr>(host.address());
        onboard.dstStride = shape.numBytes;
        onboard.srcStride = shape.numBytes;
        onboard.bytesPerPage = static_cast<uint32_t>(shape.numBytes);
        onboard.pairs = pairPtr;
        onboard.numPairs = static_cast<uint32_t>(shape.numTasks);
        PoolCopyArgs offload = onboard;
        offload.dstBase = onboard.srcBase;
        offload.srcBase = onboard.dstBase;

        auto timeGbPerSec = [&](auto&& issue) -> double
        {
            constexpr int kWarmup = 3;
            constexpr int kIters = 10;
            for (int i = 0; i < kWarmup; ++i)
            {
                issue();
            }
            EXPECT_EQ(cudaEventRecord(start, mStream), cudaSuccess);
            for (int i = 0; i < kIters; ++i)
            {
                issue();
            }
            EXPECT_EQ(cudaEventRecord(stop, mStream), cudaSuccess);
            EXPECT_EQ(cudaEventSynchronize(stop), cudaSuccess);
            float ms = 0.0F;
            EXPECT_EQ(cudaEventElapsedTime(&ms, start, stop), cudaSuccess);
            return static_cast<double>(total) * kIters / (static_cast<double>(ms) * 1e-3) / 1e9;
        };

        double const helperH2d = timeGbPerSec(
            [&]
            {
                EXPECT_EQ(copyHostToDevice(
                              hostToDevice, static_cast<ssize_t>(shape.numBytes), reinterpret_cast<CUstream>(mStream)),
                    CUDA_SUCCESS);
            });
        double const helperD2h = timeGbPerSec(
            [&]
            {
                EXPECT_EQ(copyDeviceToHost(
                              deviceToHost, static_cast<ssize_t>(shape.numBytes), reinterpret_cast<CUstream>(mStream)),
                    CUDA_SUCCESS);
            });
        double const copierH2d
            = timeGbPerSec([&] { copier.launch(onboard, CopyDirection::kH2D, reinterpret_cast<CUstream>(mStream)); });
        double const copierD2h
            = timeGbPerSec([&] { copier.launch(offload, CopyDirection::kD2H, reinterpret_cast<CUstream>(mStream)); });

        // Correctness of the copier path on this layout: device <- host (0x3C) after a 0xEE fill.
        ASSERT_EQ(cudaMemsetAsync(device.get(), 0xEE, total, mStream), cudaSuccess);
        copier.launch(onboard, CopyDirection::kH2D, reinterpret_cast<CUstream>(mStream));
        std::vector<unsigned char> readBack(total);
        ASSERT_EQ(cudaMemcpyAsync(readBack.data(), device.get(), total, cudaMemcpyDeviceToHost, mStream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        bool const copierOk = std::all_of(readBack.begin(), readBack.end(), [](unsigned char v) { return v == 0x3C; });

        std::cout << "PathComparisonInfo: " << shape.name << " (" << shape.numTasks << " x " << shape.numBytes
                  << " B): helpers[" << (useCopyEngineForHostCopy(shape.numBytes) ? "copy-engine" : "kernel")
                  << "] h2d " << helperH2d << " GB/s, d2h " << helperD2h << " GB/s | BatchedPageCopier["
                  << (copier.kernelEnabled() ? "kernel" : "copy-engine") << "] h2d " << copierH2d << " GB/s, d2h "
                  << copierD2h << " GB/s, bytes " << (copierOk ? "ok" : "MISMATCH") << std::endl;
        EXPECT_TRUE(copierOk) << shape.name;
    }

    ASSERT_EQ(cudaEventDestroy(start), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(stop), cudaSuccess);
}

// Occupies every thread slot of the GPU for `cycles` clock ticks per block, launched back to back on its own stream, to
// stand in for a compute phase that leaves no room for other CTAs.
__global__ void spinKernel(long long cycles, int* sink)
{
    long long const start = clock64();
    int acc = 0;
    while (clock64() - start < cycles)
    {
        acc += 1;
    }
    if (acc == -1)
    {
        *sink = acc;
    }
}

// TEMPORARY BENCHMARK (not for the PR): the same comparison while the SMs are saturated by a spin kernel, with the copy
// stream at default and at the highest priority. Copy engines need no SM; the copier's CTAs must wait for a slot.
TEST_F(KvCacheManagerV2HostCopyTest, PathComparisonUnderLoadInfo)
{
    ScopedEnv prefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "0");
    ScopedEnv thp("TLLM_KV_CACHE_MANAGER_V2_THP", "1");

    BatchedPageCopier const copier;
    int const smCount = copier.topology().smCount;
    int lowest = 0;
    int highest = 0;
    ASSERT_EQ(cudaDeviceGetStreamPriorityRange(&lowest, &highest), cudaSuccess);
    cudaStream_t busy = nullptr;
    cudaStream_t highPriority = nullptr;
    ASSERT_EQ(cudaStreamCreateWithPriority(&busy, cudaStreamNonBlocking, lowest), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithPriority(&highPriority, cudaStreamNonBlocking, highest), cudaSuccess);
    CudaAllocation sink = allocateCuda(sizeof(int));
    ASSERT_NE(sink.get(), nullptr);
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

    // Calibrate the spin: clock ticks per millisecond on this part.
    constexpr long long kCalibrationCycles = 2'000'000;
    ASSERT_EQ(cudaEventRecord(start, busy), cudaSuccess);
    spinKernel<<<1, 32, 0, busy>>>(kCalibrationCycles, static_cast<int*>(sink.get()));
    ASSERT_EQ(cudaEventRecord(stop, busy), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);
    float calibrationMs = 0.0F;
    ASSERT_EQ(cudaEventElapsedTime(&calibrationMs, start, stop), cudaSuccess);
    double const cyclesPerMs = kCalibrationCycles / static_cast<double>(calibrationMs);
    std::cout << "PathComparisonUnderLoadInfo: smCount=" << smCount << " priorities " << lowest << ".." << highest
              << " cyclesPerMs=" << cyclesPerMs << " copier ctas(d2h/h2d)=" << copier.config(CopyDirection::kD2H).ctas
              << "/" << copier.config(CopyDirection::kH2D).ctas << std::endl;

    struct Shape
    {
        size_t numBytes;
        size_t numTasks;
        char const* name;
    };

    for (Shape const shape : {Shape{kSsmStateBytes, 1, "ssm-state x1"}, Shape{kConvStateBytes, 12, "conv-state x12"}})
    {
        size_t const total = shape.numBytes * shape.numTasks;
        HostMem host(roundUpToHostMemAlignment(total));
        CudaAllocation device = allocateCuda(total);
        ASSERT_NE(device.get(), nullptr);
        std::memset(reinterpret_cast<void*>(host.address()), 0x3C, total);
        auto const deviceBase = reinterpret_cast<MemAddress>(device.get());
        std::vector<MMTask> hostToDevice;
        std::vector<PageIndexPair> pairs;
        for (size_t i = 0; i < shape.numTasks; ++i)
        {
            hostToDevice.push_back(MMTask{deviceBase + i * shape.numBytes, host.address() + i * shape.numBytes});
            pairs.push_back(PageIndexPair{static_cast<int32_t>(i), static_cast<int32_t>(i)});
        }
        CudaAllocation devicePairs;
        PageIndexPair const* pairPtr = pairs.data();
        if (copier.pageIndexLocation() == PageIndexLocation::kDevice)
        {
            devicePairs = allocateCuda(pairs.size() * sizeof(PageIndexPair));
            ASSERT_NE(devicePairs.get(), nullptr);
            ASSERT_EQ(cudaMemcpy(devicePairs.get(), pairs.data(), pairs.size() * sizeof(PageIndexPair),
                          cudaMemcpyHostToDevice),
                cudaSuccess);
            pairPtr = static_cast<PageIndexPair const*>(devicePairs.get());
        }
        PoolCopyArgs onboard{};
        onboard.dstBase = static_cast<CUdeviceptr>(deviceBase);
        onboard.srcBase = static_cast<CUdeviceptr>(host.address());
        onboard.dstStride = shape.numBytes;
        onboard.srcStride = shape.numBytes;
        onboard.bytesPerPage = static_cast<uint32_t>(shape.numBytes);
        onboard.pairs = pairPtr;
        onboard.numPairs = static_cast<uint32_t>(shape.numTasks);

        // Milliseconds per host->device copy on `copyStream` while `busy` runs blocks of `blockMs` each, every thread
        // slot of every SM taken (8 x 256 threads per SM). blockMs 0 means no load.
        auto msPerCopy = [&](double blockMs, cudaStream_t copyStream, auto&& issue) -> double
        {
            constexpr int kWarmup = 3;
            constexpr int kIters = 10;
            if (blockMs > 0.0)
            {
                auto const cycles = static_cast<long long>(blockMs * cyclesPerMs);
                int const launches = static_cast<int>(400.0 / blockMs) + 1; // about 400 ms of load
                for (int i = 0; i < launches; ++i)
                {
                    spinKernel<<<smCount * 8, 256, 0, busy>>>(cycles, static_cast<int*>(sink.get()));
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10)); // let the load establish
            }
            for (int i = 0; i < kWarmup; ++i)
            {
                issue(copyStream);
            }
            EXPECT_EQ(cudaEventRecord(start, copyStream), cudaSuccess);
            for (int i = 0; i < kIters; ++i)
            {
                issue(copyStream);
            }
            EXPECT_EQ(cudaEventRecord(stop, copyStream), cudaSuccess);
            EXPECT_EQ(cudaEventSynchronize(stop), cudaSuccess);
            float ms = 0.0F;
            EXPECT_EQ(cudaEventElapsedTime(&ms, start, stop), cudaSuccess);
            EXPECT_EQ(cudaStreamSynchronize(busy), cudaSuccess); // drain the remaining load before the next case
            return static_cast<double>(ms) / kIters;
        };
        auto issueHelper = [&](cudaStream_t s)
        {
            EXPECT_EQ(
                copyHostToDevice(hostToDevice, static_cast<ssize_t>(shape.numBytes), reinterpret_cast<CUstream>(s)),
                CUDA_SUCCESS);
        };
        auto issueCopier
            = [&](cudaStream_t s) { copier.launch(onboard, CopyDirection::kH2D, reinterpret_cast<CUstream>(s)); };

        for (double const blockMs : {0.0, 0.1, 5.0})
        {
            for (int prio = 0; prio < 2; ++prio)
            {
                cudaStream_t const copyStream = prio == 0 ? mStream : highPriority;
                double const helperMs = msPerCopy(blockMs, copyStream, issueHelper);
                double const copierMs = msPerCopy(blockMs, copyStream, issueCopier);
                std::cout << "PathComparisonUnderLoadInfo: " << shape.name << " h2d, load blocks " << blockMs
                          << " ms, copy stream " << (prio == 0 ? "default" : "highest") << " priority: helpers["
                          << (useCopyEngineForHostCopy(shape.numBytes) ? "copy-engine" : "kernel") << "] " << helperMs
                          << " ms/copy (" << total / (helperMs * 1e-3) / 1e9 << " GB/s) | BatchedPageCopier["
                          << (copier.kernelEnabled() ? "kernel" : "copy-engine") << "] " << copierMs << " ms/copy ("
                          << total / (copierMs * 1e-3) / 1e9 << " GB/s)" << std::endl;
            }
        }
    }

    ASSERT_EQ(cudaEventDestroy(start), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(stop), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(busy), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(highPriority), cudaSuccess);
}

} // namespace
