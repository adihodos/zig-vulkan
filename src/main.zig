const std = @import("std");
const builtin = @import("builtin");
const sdl = @import("sdl2");
const gfx = @import("gfx.zig");

const Pixel = struct {
    r: u8,
    g: u8,
    b: u8,

    const Self = @This();
};

fn write_ppm_file(file_path: []const u8, pixels: []const Pixel, width: u32, height: u32) !void {
    var out_file = try std.fs.createFileAbsolute(file_path, .{ .truncate = true });
    defer {
        out_file.close();
    }

    var bw = std.io.BufferedWriter(4096, @TypeOf(out_file.writer())){ .unbuffered_writer = out_file.writer() };
    defer bw.flush() catch {};

    try bw.writer().print("P3\n{d} {d} 255\n", .{ width, height });

    var y: u32 = 0;
    while (y < height) : (y += 1) {
        var x: u32 = 0;
        while (x < width) : (x += 1) {
            const p = pixels[y * width + x];
            try bw.writer().print("{d} {d} {d}\n", .{ p.r, p.g, p.b });
        }
    }
}

const Cf32 = std.math.complex.Complex(f32);

const FractalFunc = enum(u8) {
    Quadratic,
    Cubic,
    Cosine,
    Sine,
};

const FractalFuncType = *const fn (x: u32, y: u32, z: Cf32, c: Cf32, p: *const FractalParameters) FractalIterationResult;

const FractalParameters = struct {
    width: u32,
    height: u32,
    max_iterations: u32,
    escape_radius: f32,
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
    c: Cf32,
    coloring: FractalColoring,
    fractal_func: FractalFuncType,

    const Self = @This();

    pub fn format(self: *const Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        try writer.print(
            "FractalParameters = {{\n\t.width = {d}, .height = {d}, .max_iterations = {d}, .escape_radius = {d:.2},",
            .{ self.width, self.height, self.max_iterations, self.escape_radius },
        );

        try writer.print(
            "\n\t.xmin = {d:.2}, .xmax = {d:.2}\n\t.ymin = {d:.2}, .ymax = {d:.2}\n\t.c = Complex = {{ .re = {d:.4}, .im = {d:.4} }}",
            .{ self.xmin, self.xmax, self.ymin, self.ymax, self.c.re, self.c.im },
        );

        try writer.print("\n\t.coloring = {s}", .{@tagName(self.coloring)});
        try writer.print("\n}}", .{});
    }
};

const FractalIterationResult = struct {
    x: u32,
    y: u32,
    iterations: u32,
    c: std.math.complex.Complex(f32),
};

fn screen_coord_to_complex(
    px: f32,
    py: f32,
    dxmin: f32,
    dxmax: f32,
    dymin: f32,
    dymax: f32,
    screen_width: f32,
    screen_height: f32,
) Cf32 {
    return Cf32{
        .re = (px / screen_width) * (dxmax - dxmin) + dxmin,
        .im = (py / screen_height) * (dymax - dymin) + dymin,
    };
}

fn julia_quadratic(x: u32, y: u32, z_init: Cf32, c: Cf32, params: *const FractalParameters) FractalIterationResult {
    var z = z_init;
    var iterations: u32 = 0;
    while (((z.re * z.re + z.im * z.im) <= (params.escape_radius * params.escape_radius)) and (iterations < params.max_iterations)) : (iterations += 1) {
        z = (z.mul(z)).add(c);
    }

    var extra_iter: u32 = 0;
    while (extra_iter < 2) : (extra_iter += 1) {
        z = (z.mul(z)).add(c);
    }

    return FractalIterationResult{
        .x = x,
        .y = y,
        .c = z,
        .iterations = iterations,
    };
}

fn julia_cubic(x: u32, y: u32, z_init: Cf32, c: Cf32, params: *const FractalParameters) FractalIterationResult {
    var z = z_init;
    var iterations: u32 = 0;

    while (((z.re * z.re + z.im * z.im) <= (params.escape_radius * params.escape_radius)) and (iterations < params.max_iterations)) : (iterations += 1) {
        z = (z.mul(z).mul(z)).add(c);
    }

    var extra_iter: u32 = 0;
    while (extra_iter < 2) : (extra_iter += 1) {
        z = (z.mul(z).mul(z)).add(c);
    }

    return FractalIterationResult{
        .x = x,
        .y = y,
        .c = z,
        .iterations = iterations,
    };
}

fn julia_cosine(x: u32, y: u32, z_init: Cf32, c: Cf32, params: *const FractalParameters) FractalIterationResult {
    var z = z_init;

    var iterations: u32 = 0;
    while ((@abs(z.im) < 50.0) and (iterations < params.max_iterations)) : (iterations += 1) {
        z = c.mul(std.math.complex.cos(z));
    }

    var extra_iter: u32 = 0;
    while (extra_iter < 2) : (extra_iter += 1) {
        z = c.mul(std.math.complex.cos(z));
    }

    return FractalIterationResult{
        .x = x,
        .y = y,
        .c = z,
        .iterations = iterations,
    };
}

fn julia_sine(x: u32, y: u32, z_init: Cf32, c: Cf32, params: *const FractalParameters) FractalIterationResult {
    var z = z_init;

    var iterations: u32 = 0;
    while ((@abs(z.im) < 50.0) and (iterations < params.max_iterations)) : (iterations += 1) {
        z = c.mul(std.math.complex.sin(z));
    }

    var extra_iter: u32 = 0;
    while (extra_iter < 2) : (extra_iter += 1) {
        z = c.mul(std.math.complex.sin(z));
    }

    return FractalIterationResult{
        .x = x,
        .y = y,
        .c = z,
        .iterations = iterations,
    };
}

const WorkPackage = struct {
    xmin: u32,
    xmax: u32,
    ymin: u32,
    ymax: u32,
};

const WorkerParams = struct {
    work_queue: std.ArrayList(WorkPackage),
    lock: std.Thread.Mutex,
    pixels: []Pixel,
};

fn julia(alloc: std.mem.Allocator, params: *const FractalParameters) ![]FractalIterationResult {
    var pixels = try alloc.alloc(FractalIterationResult, params.width * params.height);

    var y: u32 = 0;
    while (y < params.height) : (y += 1) {
        var x: u32 = 0;

        while (x < params.width) : (x += 1) {
            const z = screen_coord_to_complex(
                @floatFromInt(x),
                @floatFromInt(y),
                params.xmin,
                params.xmax,
                params.ymin,
                params.ymax,
                @floatFromInt(params.width),
                @floatFromInt(params.height),
            );

            pixels[y * params.width + x] = params.fractal_func(x, y, z, params.c, params);
        }
    }

    return pixels;
}

fn julia_mt(worker: *WorkerParams, params: *const FractalParameters) void {
    worker_loop: while (true) {
        const work_package = get_work_package_or_quit: {
            worker.lock.lock();
            defer worker.lock.unlock();

            if (worker.work_queue.items.len == 0) {
                break :worker_loop;
            }

            break :get_work_package_or_quit worker.work_queue.pop();
        };

        var y: u32 = work_package.ymin;
        while (y < work_package.ymax) : (y += 1) {
            var x: u32 = work_package.xmin;

            while (x < work_package.xmax) : (x += 1) {
                const z = screen_coord_to_complex(
                    @floatFromInt(x),
                    @floatFromInt(y),
                    params.xmin,
                    params.xmax,
                    params.ymin,
                    params.ymax,
                    @floatFromInt(params.width),
                    @floatFromInt(params.height),
                );

                const iteration_result = params.fractal_func(x, y, z, params.c, params);
                worker.pixels[y * params.width + x] = switch (params.coloring) {
                    FractalColoring.BlackWhite => color_simple(&iteration_result, params),
                    FractalColoring.Smooth => color_smooth(&iteration_result, params),
                    FractalColoring.Logarithmic => color_logarithmic(&iteration_result, params),
                };
            }
        }
    }
}

fn color_simple(c: *const FractalIterationResult, p: *const FractalParameters) Pixel {
    const color = std.math.clamp(@as(f32, @floatFromInt(c.iterations)) / @as(f32, @floatFromInt(p.max_iterations)) * 255.0, 0.0, 255.0);
    return Pixel{
        .r = @as(u8, @intFromFloat(color)),
        .g = @as(u8, @intFromFloat(color)),
        .b = @as(u8, @intFromFloat(color)),
    };
}

fn color_smooth(c: *const FractalIterationResult, p: *const FractalParameters) Pixel {
    const t = std.math.clamp(@as(f32, @floatFromInt(c.iterations)) / @as(f32, @floatFromInt(p.max_iterations)), 0.0, 1.0);
    const u = 1.0 - t;
    return Pixel{
        .r = @as(u8, @intFromFloat(std.math.clamp(9.0 * u * t * t * t * 255.0, 0.0, 255.0))),
        .g = @as(u8, @intFromFloat(std.math.clamp(15.0 * u * u * t * t * 255.0, 0.0, 255.0))),
        .b = @as(u8, @intFromFloat(std.math.clamp(8.5 * u * u * u * t * 255.0, 0.0, 255.0))),
    };
}

fn color_logarithmic(c: *const FractalIterationResult, p: *const FractalParameters) Pixel {
    const n = @as(f32, @floatFromInt(c.iterations)) + 1.0 - std.math.log2(std.math.log2(c.c.magnitude()) / std.math.log2(p.escape_radius));

    return Pixel{
        .r = @as(u8, @intFromFloat(std.math.clamp(1.0 - std.math.cos(0.025 * n), 0.0, 1.0) * 255.0)),
        .g = @as(u8, @intFromFloat(std.math.clamp(1.0 - std.math.cos(0.080 * n), 0.0, 1.0) * 255.0)),
        .b = @as(u8, @intFromFloat(std.math.clamp(1.0 - std.math.cos(0.120 * n), 0.0, 1.0) * 255.0)),
    };
}

const FractalColoring = enum(u8) {
    BlackWhite,
    Smooth,
    Logarithmic,
};

const FractalError = error{PepegaArgs};

const WorkerConfiguration = struct {
    workers: u32,
    shuffle_packages: bool,
};

// var g_scratch_buffer: [8192 * 8192]raylib.Color = undefined;

var g_scratch_buffer: [8192 * 8192]u8 = undefined;

const PhysicalDeviceData = struct {
    device: gfx.VkPhysicalDevice,
    props: gfx.VkPhysicalDeviceProperties2,
    props_vk11: gfx.VkPhysicalDeviceVulkan11Properties,
    props_vk12: gfx.VkPhysicalDeviceVulkan12Properties,
    props_vk13: gfx.VkPhysicalDeviceVulkan13Properties,
    features: gfx.VkPhysicalDeviceFeatures2,
    features_vk11: gfx.VkPhysicalDeviceVulkan11Features,
    features_vk12: gfx.VkPhysicalDeviceVulkan12Features,
    features_vk13: gfx.VkPhysicalDeviceVulkan13Features,
    memory: gfx.VkPhysicalDeviceMemoryProperties,

    pub fn create(device: gfx.VkPhysicalDevice) PhysicalDeviceData {
        var pdd: PhysicalDeviceData = PhysicalDeviceData{
            .device = device,
            .props = gfx.VkPhysicalDeviceProperties2{
                .sType = gfx.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                .pNext = null,
            },
            .props_vk11 = gfx.VkPhysicalDeviceVulkan11Properties{
                .sType = gfx.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES,
                .pNext = null,
            },
            .props_vk12 = gfx.VkPhysicalDeviceVulkan12Properties{
                .sType = gfx.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES,
                .pNext = null,
            },
            .props_vk13 = gfx.VkPhysicalDeviceVulkan13Properties{
                .sType = gfx.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES,
                .pNext = null,
            },
            .features = gfx.VkPhysicalDeviceFeatures2{
                .sType = gfx.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                .pNext = null,
            },
            .features_vk11 = gfx.VkPhysicalDeviceVulkan11Features{
                .sType = gfx.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                .pNext = null,
            },
            .features_vk12 = gfx.VkPhysicalDeviceVulkan12Features{
                .sType = gfx.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                .pNext = null,
            },
            .features_vk13 = gfx.VkPhysicalDeviceVulkan13Features{
                .sType = gfx.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                .pNext = null,
            },
            .memory = undefined,
        };

        pdd.props.pNext = &pdd.props_vk11;
        pdd.props_vk11.pNext = &pdd.props_vk12;
        pdd.props_vk12.pNext = &pdd.props_vk13;

        pdd.features.pNext = &pdd.features_vk11;
        pdd.features_vk11.pNext = &pdd.features_vk12;
        pdd.features_vk12.pNext = &pdd.features_vk13;

        gfx.vkGetPhysicalDeviceProperties2(pdd.device, &pdd.props);
        gfx.vkGetPhysicalDeviceFeatures2(pdd.device, &pdd.features);
        gfx.vkGetPhysicalDeviceMemoryProperties(pdd.device, &pdd.memory);
        return pdd;
    }

    pub fn check_feature_support(self: *const PhysicalDeviceData) bool {
        return self.features_vk11.shaderDrawParameters != 0 and
            self.features_vk12.descriptorIndexing != 0 and
            self.features_vk12.descriptorBindingPartiallyBound != 0;
    }
};

fn vk_dbg_utils_msg_callback(
    severity: gfx.VkDebugUtilsMessageSeverityFlagBitsEXT,
    msg_types: gfx.VkDebugUtilsMessageTypeFlagBitsEXT,
    cb_data: [*c]const gfx.VkDebugUtilsMessengerCallbackDataEXT,
    user: ?*anyopaque,
) callconv(.C) gfx.VkBool32 {
    _ = msg_types;
    _ = user;

    if ((severity & gfx.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) {
        std.log.warn("\n[Vulkan] {d}:{s}:{s}", .{ cb_data[0].messageIdNumber, cb_data[0].pMessageIdName, cb_data[0].pMessage });
    } else if ((severity & gfx.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
        std.log.err("\n[Vulkan] {d}:{s}:{s}", .{ cb_data[0].messageIdNumber, cb_data[0].pMessageIdName, cb_data[0].pMessage });
    } else {
        std.log.info("\n[Vulkan] {d}:{s}:{s}", .{ cb_data[0].messageIdNumber, cb_data[0].pMessageIdName, cb_data[0].pMessage });
    }

    return gfx.VK_FALSE;
}

fn get_physical_device(vkinst: gfx.VkInstance, surface: gfx.VkSurfaceKHR, allocator: std.mem.Allocator) ?gfx.VkPhysicalDevice {
    const phys_devices = enum_physical_devices: {
        var count: u32 = 0;
        _ = gfx.vkEnumeratePhysicalDevices(vkinst, &count, null);
        if (count == 0) {
            return null;
        }

        var phys_devices = std.ArrayList(gfx.VkPhysicalDevice).initCapacity(allocator, count) catch {
            return null;
        };
        phys_devices.resize(count) catch {
            return null;
        };

        _ = gfx.vkEnumeratePhysicalDevices(vkinst, &count, phys_devices.items.ptr);
        break :enum_physical_devices phys_devices;
    };
    defer {
        phys_devices.deinit();
    }

    // var phys_dev_props = std.ArrayList(PhysicalDeviceData).initCapacity(allocator, phys_devices.items.len) catch {
    //     return null;
    // };

    const pdd = find_best_physical_device: for (phys_devices.items) |pd| {
        const pdd = PhysicalDeviceData.create(pd);
        std.log.info("Phys device {s}, device type {d}", .{ pdd.props.properties.deviceName, pdd.props.properties.deviceType });

        if (pdd.props.properties.deviceType != gfx.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU and
            pdd.props.properties.deviceType != gfx.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            std.log.info("Rejecting device, not a true GPU", .{});
            continue;
        }

        std.log.info(
            ":::Device properties:::\n{}\n\n:::Device features:::\n{}:::Memory properties:::\n{}::Vk11 Features:::\n{}\n:::Vk12 Features:::\n{}",
            .{
                pdd.props.properties,
                pdd.features.features,
                pdd.memory,
                pdd.features_vk11,
                pdd.features_vk12,
            },
        );

        if (!pdd.check_feature_support()) {
            std.log.info("Rejecting device {s}, does not support all required features.", .{pdd.props.properties.deviceName});
            continue;
        }

        const queue_families = get_queue_families: {
            var queues: u32 = 0;
            gfx.vkGetPhysicalDeviceQueueFamilyProperties(pdd.device, &queues, null);
            var queue_props = std.ArrayList(gfx.VkQueueFamilyProperties).initCapacity(allocator, queues) catch {
                return null;
            };
            queue_props.resize(queues) catch {
                return null;
            };
            gfx.vkGetPhysicalDeviceQueueFamilyProperties(pdd.device, &queues, queue_props.items.ptr);
            break :get_queue_families queue_props;
        };
        defer {
            queue_families.deinit();
        }

        const queue_id = find_graphics_queue_index: for (queue_families.items, 0..) |qp, qidx| {
            if ((qp.queueFlags & gfx.VK_QUEUE_GRAPHICS_BIT) != 0)
                break :find_graphics_queue_index @as(u32, @intCast(qidx));
        } else {
            std.log.info("Rejecting device {s}, no queue with graphics support", .{pdd.props.properties.deviceName});
            continue;
        };

        std.log.info("Graphics queue idx {d}", .{queue_id});

        {
            var surface_supported: gfx.VkBool32 = gfx.VK_FALSE;
            const query_result = gfx.vkGetPhysicalDeviceSurfaceSupportKHR(pdd.device, queue_id, surface, &surface_supported);
            if (query_result != gfx.VK_SUCCESS) {
                std.log.err("Failed to query device for surface support, error {d}", .{query_result});
                continue;
            }
            if (surface_supported != gfx.VK_TRUE) {
                std.log.err("Rejecting device {s}, no surface support.", .{pdd.props.properties.deviceName});
                continue;
            }
        }

        const surface_formats = get_phys_device_surface_formats: {
            var fmt_count: u32 = 0;
            var query_result = gfx.vkGetPhysicalDeviceSurfaceFormatsKHR(pdd.device, surface, &fmt_count, null);

            if (query_result != gfx.VK_SUCCESS or fmt_count == 0) {
                std.log.err("Failed to query surface formats, error {d}", .{query_result});
                continue;
            }

            var surface_formats = std.ArrayList(gfx.VkSurfaceFormatKHR).initCapacity(allocator, fmt_count) catch {
                return null;
            };
            surface_formats.resize(fmt_count) catch {
                return null;
            };

            query_result = gfx.vkGetPhysicalDeviceSurfaceFormatsKHR(pdd.device, surface, &fmt_count, surface_formats.items.ptr);
            if (query_result != gfx.VK_SUCCESS or fmt_count == 0) {
                std.log.err("Failed to query surface formats, error {d}", .{query_result});
                continue;
            }

            break :get_phys_device_surface_formats surface_formats;
        };
        defer {
            surface_formats.deinit();
        }

        const required_fmts = [_]u32{
            gfx.VK_FORMAT_R8G8B8A8_UNORM,
            gfx.VK_FORMAT_R8G8B8A8_SRGB,
            gfx.VK_FORMAT_B8G8R8A8_UNORM,
            gfx.VK_FORMAT_B8G8R8A8_SRGB,
        };

        const picked_surface_fmt = pick_best_format: for (surface_formats.items) |khr_fmt| {
            std.log.info("checking device support for format {any}", .{khr_fmt});
            for (required_fmts) |req_fmt| {
                if (req_fmt == khr_fmt.format)
                    break :pick_best_format khr_fmt;
            }
        } else {
            std.log.err("Rejecting device {s}, none of the required formats are supported", .{pdd.props.properties.deviceName});
            continue;
        };

        std.log.info("Picked surface format {}", .{picked_surface_fmt});
        var surface_caps: gfx.VkSurfaceCapabilitiesKHR = undefined;
        const query_result = gfx.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pdd.device, surface, &surface_caps);
        if (query_result != gfx.VK_SUCCESS) {
            std.log.err("Failed to query surface capabilities, error 0x{X:0>8}", .{@as(u32, @bitCast(query_result))});
            continue;
        }

        break :find_best_physical_device .{
            .pdd = pdd,
            .surface = surface,
            .format = picked_surface_fmt,
            .caps = surface_caps,
        };
    } else {
        std.log.err("No suitable device present!", .{});
        return null;
    };

    std.log.info("Picked device {any}", .{pdd});
    return null;
}

const VulkanWsiCreateInfo = union(enum) {
    win32: struct {
        hwnd: std.os.windows.HWND,
        hinstance: std.os.windows.HINSTANCE,
    },
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // const allocator = gpa.allocator();
    defer {
        _ = gpa.deinit();
    }

    var fba_alloc = std.heap.FixedBufferAllocator.init(&g_scratch_buffer);
    const fba = fba_alloc.allocator();
    //
    var stdout = std.io.getStdOut().writer();
    //
    // const IMAGE_WIDTH: u32 = 1024;
    // const IMAGE_HEIGHT: u32 = 1024;
    //
    // var args = try std.process.argsWithAllocator(allocator);
    // defer args.deinit();
    //
    // const program_setup = params: {
    //     var params = FractalParameters{
    //         .width = IMAGE_WIDTH,
    //         .height = IMAGE_HEIGHT,
    //         .max_iterations = 32,
    //         .escape_radius = 2.0,
    //         .xmin = -1.0,
    //         .xmax = 1.0,
    //         .ymin = -1.0,
    //         .ymax = 1.0,
    //         .c = Cf32{ .re = 0.355, .im = 0.355 },
    //         .coloring = FractalColoring.BlackWhite,
    //         .fractal_func = &julia_quadratic,
    //     };
    //
    //     var workers_setup = WorkerConfiguration{
    //         .workers = 1,
    //         .shuffle_packages = false,
    //     };
    //
    //     _ = args.skip();
    //
    //     while (args.next()) |arg| {
    //         if (std.mem.startsWith(u8, arg, "--width=")) {
    //             params.width = std.fmt.parseInt(u32, arg[std.mem.indexOf(u8, arg, "=").? + 1 ..], 10) catch {
    //                 continue;
    //             };
    //         } else if (std.mem.startsWith(u8, arg, "--height=")) {
    //             params.height = std.fmt.parseInt(u32, arg[std.mem.indexOf(u8, arg, "=").? + 1 ..], 10) catch {
    //                 continue;
    //             };
    //         } else if (std.mem.startsWith(u8, arg, "--iterations=")) {
    //             params.max_iterations = std.fmt.parseInt(u32, arg[std.mem.indexOf(u8, arg, "=").? + 1 ..], 10) catch {
    //                 continue;
    //             };
    //         } else if (std.mem.startsWith(u8, arg, "--coloring=")) {
    //             if (std.meta.stringToEnum(FractalColoring, arg[std.mem.indexOf(u8, arg, "=").? + 1 ..])) |coloring| {
    //                 params.coloring = coloring;
    //             }
    //         } else if (std.mem.startsWith(u8, arg, "--origin=")) {
    //             const eqpos = std.mem.indexOf(u8, arg, "=").? + 1;
    //             const cstart = arg[eqpos..];
    //             const comma_pos = std.mem.indexOf(u8, cstart, "x").?;
    //             params.c = Cf32{
    //                 .re = std.fmt.parseFloat(f32, cstart[0..comma_pos]) catch {
    //                     continue;
    //                 },
    //                 .im = std.fmt.parseFloat(f32, cstart[comma_pos + 1 ..]) catch {
    //                     continue;
    //                 },
    //             };
    //         } else if (std.mem.startsWith(u8, arg, "--func=")) {
    //             if (std.meta.stringToEnum(FractalFunc, arg[std.mem.indexOf(u8, arg, "=").? + 1 ..])) |fractal_fn| {
    //                 params.fractal_func =
    //                     switch (fractal_fn) {
    //                     .Cubic => &julia_cubic,
    //                     .Quadratic => &julia_quadratic,
    //                     .Cosine => &julia_cosine,
    //                     .Sine => &julia_sine,
    //                 };
    //             }
    //         } else if (std.mem.startsWith(u8, arg, "--threads=")) {
    //             workers_setup.workers = blk: {
    //                 const cpu_count: u32 = @intCast(std.Thread.getCpuCount() catch 1);
    //                 const wks = std.fmt.parseInt(u32, arg[std.mem.indexOf(u8, arg, "=").? + 1 ..], 10) catch cpu_count;
    //                 if (wks == 0) {
    //                     break :blk cpu_count;
    //                 } else {
    //                     break :blk @min(wks, cpu_count * 4);
    //                 }
    //             };
    //         } else if (std.mem.startsWith(u8, arg, "--shufle")) {
    //             workers_setup.shuffle_packages = true;
    //         } else {
    //             try stdout.print("Unrecognized argument! {s}\n", .{arg});
    //         }
    //     }
    //     break :params .{ params, workers_setup };
    // };
    //
    // const params = program_setup[0];
    // const work_cfg = program_setup[1];
    //
    // try stdout.print("Program parameters: {s}, worker threads {d}\n", .{ params, work_cfg.workers });
    //
    // var scratch_buf: [1024]u8 = undefined;
    // const dir_path = try std.fs.selfExeDirPath(&scratch_buf);
    //
    // const file_path = try std.fs.path.join(allocator, &[_][]const u8{ dir_path, "mandelbrot.ppm" });
    // defer allocator.free(file_path);
    //
    // var pixels = try allocator.alloc(Pixel, params.width * params.height);
    // defer allocator.free(pixels);
    //
    // if (work_cfg.workers == 1) {
    //     var timer = try std.time.Timer.start();
    //     const julia_result = try julia(allocator, &params);
    //     defer allocator.free(julia_result);
    //
    //     for (julia_result, 0..) |jres, idx| {
    //         pixels[idx] = switch (params.coloring) {
    //             FractalColoring.BlackWhite => color_simple(&jres, &params),
    //             FractalColoring.Smooth => color_smooth(&jres, &params),
    //             FractalColoring.Logarithmic => color_logarithmic(&jres, &params),
    //         };
    //     }
    //     const elapsed_ns = timer.lap();
    //     try stdout.print(
    //         "Render time (ST): {d:.2}h :: {d:.2}m :: {d:.2}s\n",
    //         .{
    //             @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(std.time.ns_per_hour)),
    //             @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(std.time.ns_per_min)),
    //             @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(std.time.ns_per_s)),
    //         },
    //     );
    // } else {
    //     var timer = try std.time.Timer.start();
    //     const worker_block_size: u32 = 16;
    //     std.debug.assert(params.width % worker_block_size == 0);
    //     std.debug.assert(params.height % worker_block_size == 0);
    //
    //     var work_queue = std.ArrayList(WorkPackage).init(allocator);
    //     defer work_queue.deinit();
    //
    //     const pkgs_x = params.width / worker_block_size;
    //     const pkgs_y = params.height / worker_block_size;
    //
    //     var wy: u32 = 0;
    //     while (wy < pkgs_y) : (wy += 1) {
    //         var wx: u32 = 0;
    //         while (wx < pkgs_x) : (wx += 1) {
    //             try work_queue.append(WorkPackage{
    //                 .xmin = wx * worker_block_size,
    //                 .xmax = (wx + 1) * worker_block_size,
    //                 .ymin = wy * worker_block_size,
    //                 .ymax = (wy + 1) * worker_block_size,
    //             });
    //         }
    //     }
    //
    //     if (work_cfg.shuffle_packages) {
    //         try stdout.print("Shuffling work packages.\n", .{});
    //
    //         var seed: u64 = undefined;
    //         try std.posix.getrandom(std.mem.asBytes(&seed));
    //         var rng = std.Random.DefaultPrng.init(seed);
    //         std.rand.shuffle(rng.random(), WorkPackage, work_queue.items);
    //     }
    //
    //     try stdout.print("Work packages count : {d}\n", .{work_queue.items.len});
    //
    //     var worker_threads = std.ArrayList(std.Thread).init(allocator);
    //     defer worker_threads.deinit();
    //
    //     var worker_data = WorkerParams{
    //         .pixels = pixels,
    //         .work_queue = work_queue,
    //         .lock = std.Thread.Mutex{},
    //     };
    //
    //     {
    //         var i: u32 = 0;
    //         while (i < work_cfg.workers) : (i += 1) {
    //             try worker_threads.append(try std.Thread.spawn(.{}, julia_mt, .{ &worker_data, &params }));
    //         }
    //     }
    //
    //     for (worker_threads.items) |wk| {
    //         wk.join();
    //     }
    //
    //     const elapsed_ns = timer.lap();
    //     try stdout.print(
    //         "Render time (MT - {d} workers): {d:.2}h :: {d:.2}m :: {d:.2}s\n",
    //         .{
    //             work_cfg.workers,
    //             @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(std.time.ns_per_hour)),
    //             @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(std.time.ns_per_min)),
    //             @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(std.time.ns_per_s)),
    //         },
    //     );
    // }
    //
    // try write_ppm_file(file_path, pixels, params.width, params.height);
    if (sdl.SDL_Init(sdl.SDL_INIT_VIDEO | sdl.SDL_INIT_EVENTS) != 0)
        @panic("Failed to initialize SDL");

    defer {
        sdl.SDL_Quit();
    }

    std.log.info("SDL initialized", .{});
    const window = sdl.SDL_CreateWindow(
        "Zig+Vulkan->still doin ur mom!",
        sdl.SDL_WINDOWPOS_CENTERED,
        sdl.SDL_WINDOWPOS_CENTERED,
        1600,
        1200,
        sdl.SDL_WINDOW_BORDERLESS | sdl.SDL_WINDOW_ALLOW_HIGHDPI,
    ) orelse {
        @panic("Failed to create SDL window!");
    };

    var wmi: sdl.SDL_SysWMInfo = undefined;
    wmi.version = sdl.SDL_VERSION;
    _ = sdl.SDL_GetWindowWMInfo(window, &wmi);
    //     sdl.WindowPosition{ .absolute = 0 },
    //     sdl.WindowPosition{ .absolute = 0 },
    //     1600,
    //     1200,
    //     .{
    //         .resizable = true,
    //         .borderless = true,
    //         .allow_high_dpi = true,
    //     },
    // );
    //
    // defer {
    //     window.destroy();
    // }
    //
    std.log.info("SDL Window created, handle : {?}", .{wmi});

    // const wsi_info = switch (builtin.os.tag) {
    //     .windows => VulkanWsiCreateInfo{
    //         .win32 = .{
    //             .hwnd = wminfo.u.win.window,
    //             .hinstance = wminfo.u.win.hinstance,
    //         },
    //     },
    //     else => @panic("Not implemented for this platform !!"),
    // };

    const vkinstance = create_vk_instance: {
        var exts: u32 = 0;
        _ = gfx.vkEnumerateInstanceExtensionProperties(null, &exts, null);
        var exts_names = try std.ArrayList(gfx.VkExtensionProperties).initCapacity(fba, exts);
        defer {
            exts_names.deinit();
        }
        try exts_names.resize(exts);
        _ = gfx.vkEnumerateInstanceExtensionProperties(null, &exts, @ptrCast(exts_names.items.ptr));

        for (exts_names.items) |ext_prop| {
            try stdout.print("\nExtension {s} -> {d}", .{ ext_prop.extensionName, ext_prop.specVersion });
        }

        var layes: u32 = 0;
        _ = gfx.vkEnumerateInstanceLayerProperties(&layes, null);
        var layer_props = try std.ArrayList(gfx.VkLayerProperties).initCapacity(fba, layes);
        defer {
            layer_props.deinit();
        }
        try layer_props.resize(layes);
        _ = gfx.vkEnumerateInstanceLayerProperties(&layes, layer_props.items.ptr);

        for (layer_props.items) |layer| {
            try stdout.print("\nLayer {s} ({s}) : {d}", .{ layer.layerName, layer.description, layer.specVersion });
        }

        const enabled_layers = [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"};
        const enabled_extensions = [_][*:0]const u8{
            "VK_KHR_surface",
            if (builtin.os.tag == .windows) "VK_KHR_win32_surface" else "whatever",
            "VK_EXT_debug_utils",
        };

        const dbg_utils_create_info = gfx.VkDebugUtilsMessengerCreateInfoEXT{
            .sType = gfx.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .pNext = null,
            .flags = 0,
            .messageSeverity = gfx.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | gfx.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
            .messageType = gfx.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | gfx.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | gfx.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = vk_dbg_utils_msg_callback,
            .pUserData = null,
        };

        const app_info = gfx.VkApplicationInfo{
            .sType = gfx.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = null,
            .pApplicationName = "doing_ur_mom_with_vulkan_from_zig",
            .applicationVersion = gfx.VK_MAKE_API_VERSION(1, 0, 0, 0),
            .pEngineName = "pepega_engine",
            .engineVersion = gfx.VK_MAKE_API_VERSION(1, 0, 0, 0),
            .apiVersion = gfx.VK_API_VERSION_1_3,
        };

        const inst_create_info = gfx.VkInstanceCreateInfo{
            .sType = gfx.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = &dbg_utils_create_info,
            .flags = 0,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = enabled_layers.len,
            .ppEnabledLayerNames = &enabled_layers,
            .enabledExtensionCount = enabled_extensions.len,
            .ppEnabledExtensionNames = &enabled_extensions,
        };

        var instance: gfx.VkInstance = null;
        const result = gfx.vkCreateInstance(&inst_create_info, null, &instance);
        if (result != gfx.VK_SUCCESS or instance == null) {
            try stdout.print("Failed to create Vulkan instance! error {x}", .{result});
            return;
        }

        break :create_vk_instance instance.?;
    };

    const surface_khr = create_vulkan_surface(vkinstance, window);
    _ = get_physical_device(vkinstance, surface_khr, fba);

    // var renderer = try sdl.createRenderer(window, null, .{ .accelerated = true });
    // defer {
    //     renderer.destroy();
    // }
    //
    // event_loop: while (true) {
    //     if (sdl.pollEvent()) |event| {
    //         switch (event) {
    //             .key_down => |keydown| {
    //                 if (keydown.keycode == sdl.Keycode.escape) {
    //                     break :event_loop;
    //                 }
    //             },
    //             else => {},
    //         }
    //     }
    //
    //     try renderer.setColor(sdl.Color.green);
    //     try renderer.clear();
    //     renderer.present();
    // }
    //
    // raylib.SetConfigFlags(raylib.FLAG_WINDOW_UNDECORATED | raylib.FLAG_VSYNC_HINT);
    // raylib.InitWindow(1600, 1200, "Doing UR mom");
    // defer raylib.CloseWindow();
    //
    // const mon = raylib.GetCurrentMonitor();
    // const mw = raylib.GetMonitorWidth(mon);
    // const mh = raylib.GetMonitorHeight(mon);
    // try stdout.print("monitor {d} - [{d} x {d}]", .{ mon, mw, mh });
    //
    // const sw = raylib.GetScreenWidth();
    // const sh = raylib.GetScreenHeight();
    // const hw: u32 = @intCast(raylib.GetRenderWidth());
    // const hh: u32 = @intCast(raylib.GetRenderHeight());
    //
    // try stdout.print("monitor [{d} x {d}], screen [{d} x {d}], render [{d} x {d}]", .{ mw, mh, sw, sh, hw, hh });
    //
    // {
    //     var y: u32 = 0;
    //     const fwidth = 1.0 / @as(f32, @floatFromInt(hw));
    //     const fheight = 1.0 / @as(f32, @floatFromInt(hh));
    //
    //     while (y < hh) : (y += 1) {
    //         var x: u32 = 0;
    //         const g: u8 = @intFromFloat((@as(f32, @floatFromInt(y)) * fheight) * 255.0);
    //         while (x < hw) : (x += 1) {
    //             const r: u8 = @intFromFloat((@as(f32, @floatFromInt(x)) * fwidth) * 255.0);
    //             g_scratch_buffer[y * hw + x] = raylib.Color{
    //                 .r = r,
    //                 .g = g,
    //                 //.b = @intCast((@as(u32, r) + @as(u32, g)) % 255),
    //                 .b = 0,
    //                 .a = 255,
    //             };
    //         }
    //     }
    // }
    //
    // const pixels_texture = blk: {
    //     const img = raylib.GenImageColor(@intCast(hw), @intCast(hh), raylib.MAGENTA);
    //     defer raylib.UnloadImage(img);
    //     const tex = raylib.LoadTextureFromImage(img);
    //     raylib.UpdateTextureRec(
    //         tex,
    //         raylib.Rectangle{
    //             .x = 0,
    //             .y = 0,
    //             .width = @floatFromInt(hw),
    //             .height = @floatFromInt(hh),
    //         },
    //         &g_scratch_buffer,
    //     );
    //     break :blk tex;
    // };
    //
    // defer raylib.UnloadTexture(pixels_texture);
    //
    // var worker_pool: std.Thread.Pool = undefined;
    // try std.Thread.Pool.init(&worker_pool, .{ .allocator = allocator, .n_jobs = 8 });
    // defer {
    //     worker_pool.deinit();
    // }
    //
    // const params = FractalParameters{
    //     .width = 1600,
    //     .height = 1200,
    //     .max_iterations = 256,
    //     .escape_radius = 2.0,
    //     .xmin = -1.0,
    //     .xmax = 1.0,
    //     .ymin = -1.0,
    //     .ymax = 1.0,
    //     .c = Cf32{ .re = 0.355, .im = 0.355 },
    //     .coloring = FractalColoring.Smooth,
    //     .fractal_func = &julia_quadratic,
    // };
    //
    // const worker_block_size: u32 = 16;
    // std.debug.assert(params.width % worker_block_size == 0);
    // std.debug.assert(params.height % worker_block_size == 0);
    //
    // var work_queue = std.ArrayList(WorkPackage).init(allocator);
    // defer work_queue.deinit();
    //
    // const pkgs_x = params.width / worker_block_size;
    // const pkgs_y = params.height / worker_block_size;
    //
    // var wy: u32 = 0;
    // while (wy < pkgs_y) : (wy += 1) {
    //     var wx: u32 = 0;
    //     while (wx < pkgs_x) : (wx += 1) {
    //         try work_queue.append(WorkPackage{
    //             .xmin = wx * worker_block_size,
    //             .xmax = (wx + 1) * worker_block_size,
    //             .ymin = wy * worker_block_size,
    //             .ymax = (wy + 1) * worker_block_size,
    //         });
    //     }
    // }
    //
    // const shuffle_packages = true;
    // if (shuffle_packages) {
    //     try stdout.print("Shuffling work packages.\n", .{});
    //
    //     var seed: u64 = undefined;
    //     try std.posix.getrandom(std.mem.asBytes(&seed));
    //     var rng = std.Random.DefaultPrng.init(seed);
    //     std.rand.shuffle(rng.random(), WorkPackage, work_queue.items);
    // }
    //
    // try stdout.print("Work packages count : {d}\n", .{work_queue.items.len});
    //
    // var processed_packages = std.atomic.Value(u32).init(0);
    // for (work_queue.items) |pkg| {
    //     try worker_pool.spawn(julia_pool_worker, .{ params, pkg, &processed_packages });
    // }
    //
    // var combo_color_opts = std.ArrayList(u8).init(allocator);
    // defer {
    //     combo_color_opts.deinit();
    // }
    // inline for (@typeInfo(FractalColoring).Enum.fields) |fld| {
    //     stdout.print("\nEnum field {s}", .{fld.name}) catch {};
    //     try combo_color_opts.appendSlice(fld.name);
    //     try combo_color_opts.append(';');
    // }
    //
    // combo_color_opts.items[combo_color_opts.items.len - 1] = 0;
    //
    // const UiState = struct {
    //     color_opt: i32,
    //     escape_radius: f32,
    //     iterations: u32,
    // };
    // var ui_state = UiState{
    //     .color_opt = 0,
    //     .escape_radius = 2.0,
    //     .iterations = 32,
    // };
    //
    // var params_changed = false;
    // while (!raylib.WindowShouldClose()) {
    //     raylib.BeginDrawing();
    //     defer {
    //         raylib.EndDrawing();
    //     }
    //
    //     raylib.ClearBackground(raylib.RAYWHITE);
    //     raylib.UpdateTextureRec(
    //         pixels_texture,
    //         raylib.Rectangle{
    //             .x = 0,
    //             .y = 0,
    //             .width = @floatFromInt(hw),
    //             .height = @floatFromInt(hh),
    //         },
    //         &g_scratch_buffer,
    //     );
    //     raylib.DrawTexture(pixels_texture, 0, 0, raylib.WHITE);
    //
    //     const processed = processed_packages.load(std.builtin.AtomicOrder.acquire);
    //     var buf: [256]u8 = undefined;
    //     _ = try std.fmt.bufPrintZ(&buf, "Processed work items: {d} / {d}", .{ processed, work_queue.items.len });
    //     raylib.DrawText(&buf, 0, 0, 16, raylib.ORANGE);
    //
    //     if (processed != work_queue.items.len) {
    //         raylib.GuiDisable();
    //     } else {
    //         raylib.GuiEnable();
    //     }
    //
    //     if (raylib.GuiSliderBar(
    //         raylib.Rectangle{ .x = 128, .y = 128, .width = 256, .height = 24 },
    //         "Doing",
    //         "UrMom",
    //         &ui_state.escape_radius,
    //         2.0,
    //         64.0,
    //     ) != 0) {
    //         params_changed = true;
    //     }
    //
    //     var iter_count: f32 = @floatFromInt(ui_state.iterations);
    //     if (raylib.GuiSliderBar(
    //         raylib.Rectangle{ .x = 128, .y = 128 + 24 + 4, .width = 256, .height = 24 },
    //         "min",
    //         "max",
    //         &iter_count,
    //         8.0,
    //         8192.0,
    //     ) != 0) {
    //         const new_val: u32 = @intFromFloat(@ceil(iter_count));
    //
    //         if (new_val != ui_state.iterations) {
    //             params_changed = true;
    //             ui_state.iterations = new_val;
    //         }
    //     }
    //
    //     var prev_color = ui_state.color_opt;
    //     _ = raylib.GuiComboBox(
    //         raylib.Rectangle{ .x = 128, .y = 128 + 24 + 4 + 24 + 4, .width = 256, .height = 24 },
    //         combo_color_opts.items.ptr,
    //         &prev_color,
    //     );
    //
    //     if (prev_color != ui_state.color_opt) {
    //         params_changed = true;
    //         ui_state.color_opt = prev_color;
    //         stdout.print("Set coloring to {d}...", .{ui_state.color_opt}) catch {};
    //     }
    //
    //     if (raylib.GuiButton(raylib.Rectangle{ .x = 128, .y = 256, .width = 64, .height = 24 }, "Apply changes") != 0 and params_changed) {
    //         stdout.print("Applying changes...", .{}) catch {};
    //         params_changed = false;
    //         const p = FractalParameters{
    //             .width = 1600,
    //             .height = 1200,
    //             .max_iterations = ui_state.iterations,
    //             .escape_radius = ui_state.escape_radius,
    //             .xmin = -1.0,
    //             .xmax = 1.0,
    //             .ymin = -1.0,
    //             .ymax = 1.0,
    //             .c = Cf32{ .re = 0.355, .im = 0.355 },
    //             .coloring = @enumFromInt(ui_state.color_opt),
    //             .fractal_func = &julia_quadratic,
    //         };
    //
    //         processed_packages.store(0, std.builtin.AtomicOrder.seq_cst);
    //         for (work_queue.items) |pkg| {
    //             try worker_pool.spawn(julia_pool_worker, .{ p, pkg, &processed_packages });
    //         }
    //     }
    // }
}

const atomic_package_counter_t = std.atomic.Value(u32);

// fn julia_pool_worker(params: FractalParameters, pkg: WorkPackage, pkg_counter: *atomic_package_counter_t) void {
//     var y: u32 = pkg.ymin;
//     while (y < pkg.ymax) : (y += 1) {
//         var x: u32 = pkg.xmin;
//         while (x < pkg.xmax) : (x += 1) {
//             const z = screen_coord_to_complex(
//                 @floatFromInt(x),
//                 @floatFromInt(y),
//                 params.xmin,
//                 params.xmax,
//                 params.ymin,
//                 params.ymax,
//                 @floatFromInt(params.width),
//                 @floatFromInt(params.height),
//             );
//
//             const iteration_result = params.fractal_func(x, y, z, params.c, &params);
//             g_scratch_buffer[y * params.width + x] = blk: {
//                 const pixel = switch (params.coloring) {
//                     FractalColoring.BlackWhite => color_simple(&iteration_result, &params),
//                     FractalColoring.Smooth => color_smooth(&iteration_result, &params),
//                     FractalColoring.Logarithmic => color_logarithmic(&iteration_result, &params),
//                 };
//                 break :blk raylib.Color{ .r = pixel.r, .g = pixel.g, .b = pixel.b, .a = 255 };
//             };
//         }
//     }
//
//     _ = pkg_counter.fetchAdd(1, std.builtin.AtomicOrder.release);
// }
//
fn create_vulkan_surface(vkinst: gfx.VkInstance, window: *sdl.SDL_Window) gfx.VkSurfaceKHR {
    switch (builtin.os.tag) {
        .windows => {
            var wmi: sdl.SDL_SysWMInfo = undefined;
            wmi.version = sdl.SDL_VERSION;
            if (sdl.SDL_GetWindowWMInfo(window, &wmi) != sdl.SDL_TRUE) {
                @panic("Cant get platfowm window info!");
            }

            std.log.info("HINSTANCE 0x{x:0>8}, WIN 0x{x:0>8}", .{ @intFromPtr(wmi.u.win.hinstance), @intFromPtr(wmi.u.win.window) });

            const surface_create_info = gfx.VkWin32SurfaceCreateInfoKHR{
                .sType = gfx.VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                .pNext = null,
                .flags = 0,
                .hinstance = @ptrFromInt(@intFromPtr(wmi.u.win.hinstance)),
                //@alignCast(@ptrCast(wmi.u.win.hinstance)),
                .hwnd = @ptrFromInt(@intFromPtr(wmi.u.win.window)),
                // @alignCast(@ptrCast(wmi.u.win.window)),
            };

            var surface: gfx.VkSurfaceKHR = null;
            const result = gfx.vkCreateWin32SurfaceKHR(vkinst, &surface_create_info, null, &surface);
            if (result != gfx.VK_SUCCESS) {
                std.log.err("Failed to create surface, error {d}", .{result});
                @panic("Fatal error");
            }

            std.log.info("Created VkSurfaceKHR {any}", .{surface});
            return surface;
        },
        else => @panic("Not implemented for this platform!"),
    }
}
