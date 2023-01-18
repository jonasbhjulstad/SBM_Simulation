#ifndef SYCL_GRAPH_TRACY_CONFIG_HPP
#define SYCL_GRAPH_TRACY_CONFIG_HPP

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#else
#define ZoneScoped 
#define FrameMark
#define FrameMarkStart(name)
#define FrameMarkEnd(name)
#endif
#ifdef TRACY_OPENCL_ENABLE
#include <tracy/TracyOpenCL.hpp>
#else
#define TracyCLContext(ctx, device)
#define TracyCLDstroy(ctx)
#define TracyCLContextName(ctx, name, size)
#define TracyCLZone(ctx, name)
#define TracyCLSetEvent(event)
#endif

#endif