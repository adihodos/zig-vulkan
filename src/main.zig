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

const GraphicsSystemError = error{
    NoSuitableDeviceFound,
    VulkanApiError,
};

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

const PhysicalDeviceWithSurfaceData = struct {
    pdd: PhysicalDeviceData,
    surface: gfx.VkSurfaceKHR,
    surface_format: gfx.VkSurfaceFormatKHR,
    surface_caps: gfx.VkSurfaceCapabilitiesKHR,
    present_mode: gfx.VkPresentModeKHR,
    queue_family: u32,
};

fn get_physical_device(vkinst: gfx.VkInstance, surface: gfx.VkSurfaceKHR, allocator: std.mem.Allocator) !PhysicalDeviceWithSurfaceData {
    const phys_devices = enum_physical_devices: {
        var count: u32 = 0;
        var query_result = gfx.vkEnumeratePhysicalDevices(vkinst, &count, null);
        if (query_result != gfx.VK_SUCCESS or count == 0) {
            return error.VulkanApiError;
        }

        var phys_devices = try std.ArrayList(gfx.VkPhysicalDevice).initCapacity(allocator, count);
        try phys_devices.resize(count);

        query_result = gfx.vkEnumeratePhysicalDevices(vkinst, &count, phys_devices.items.ptr);
        if (query_result != gfx.VK_SUCCESS or count == 0) {
            return error.VulkanApiError;
        }
        break :enum_physical_devices phys_devices;
    };
    defer {
        phys_devices.deinit();
    }

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
            if (queues == 0)
                return error.VulkanApiError;

            var queue_props = try std.ArrayList(gfx.VkQueueFamilyProperties).initCapacity(allocator, queues);
            try queue_props.resize(queues);
            gfx.vkGetPhysicalDeviceQueueFamilyProperties(pdd.device, &queues, queue_props.items.ptr);
            if (queues == 0)
                return error.VulkanApiError;

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

            var surface_formats = try std.ArrayList(gfx.VkSurfaceFormatKHR).initCapacity(allocator, fmt_count);
            try surface_formats.resize(fmt_count);

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
        const surface_caps = get_surface_caps: {
            var surface_caps: gfx.VkSurfaceCapabilitiesKHR = undefined;
            const query_result = gfx.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pdd.device, surface, &surface_caps);
            if (query_result != gfx.VK_SUCCESS) {
                std.log.err("Failed to query surface capabilities, error 0x{X:0>8}", .{@as(u32, @bitCast(query_result))});
                continue;
            }
            break :get_surface_caps surface_caps;
        };

        const best_present_mode = pick_best_present_mode: {
            var count: u32 = 0;
            var query_result = gfx.vkGetPhysicalDeviceSurfacePresentModesKHR(pdd.device, surface, &count, null);
            if (query_result != gfx.VK_SUCCESS or count == 0) {
                std.log.err("Failed to query present modes, error 0x{x:0>8}", .{@as(u32, @bitCast(query_result))});
                continue;
            }

            var modes = try std.ArrayList(gfx.VkPresentModeKHR).initCapacity(allocator, count);
            try modes.resize(count);

            query_result = gfx.vkGetPhysicalDeviceSurfacePresentModesKHR(pdd.device, surface, &count, modes.items.ptr);
            if (query_result != gfx.VK_SUCCESS or count == 0) {
                std.log.err("Failed to query present modes, error 0x{x:0>8}", .{@as(u32, @bitCast(query_result))});
                continue;
            }
            const preferred_presentation_modes = [_]u32{
                gfx.VK_PRESENT_MODE_MAILBOX_KHR,
                gfx.VK_PRESENT_MODE_IMMEDIATE_KHR,
                gfx.VK_PRESENT_MODE_FIFO_KHR,
            };

            for (modes.items) |present_mode| {
                std.log.info("presentation mode: {}", .{present_mode});
                for (preferred_presentation_modes) |preferred_mode| {
                    if (preferred_mode == present_mode) {
                        break :pick_best_present_mode present_mode;
                    }
                }
            } else {
                std.log.err("Rejecting device {s}, no required present mode supported.", .{pdd.props.properties.deviceName});
                continue;
            }
        };

        std.log.info("Picked presentration mode: {}", .{best_present_mode});
        break :find_best_physical_device PhysicalDeviceWithSurfaceData{
            .pdd = pdd,
            .surface = surface,
            .surface_format = picked_surface_fmt,
            .surface_caps = surface_caps,
            .present_mode = best_present_mode,
            .queue_family = queue_id,
        };
    } else {
        std.log.err("No suitable device present!", .{});
        return error.NoSuitableDeviceFound;
    };

    return pdd;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // const allocator = gpa.allocator();
    defer {
        _ = gpa.deinit();
    }

    var fba_alloc = std.heap.FixedBufferAllocator.init(&g_scratch_buffer);
    const fba = fba_alloc.allocator();

    try sdl.init(.{
        .video = true,
        .events = true,
        .game_controller = true,
    });
    defer {
        sdl.quit();
    }

    const window = try sdl.createWindow(
        "Zig + SDL + Vulkan",
        .{ .centered = {} },
        .{ .centered = {} },
        1600,
        1200,
        .{ .borderless = true, .resizable = false },
    );

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
            std.log.info("\nExtension {s} -> {d}", .{ ext_prop.extensionName, ext_prop.specVersion });
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
            std.log.info("\nLayer {s} ({s}) : {d}", .{ layer.layerName, layer.description, layer.specVersion });
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
            std.log.info("Failed to create Vulkan instance! error {x}", .{result});
            return;
        }

        break :create_vk_instance instance.?;
    };

    const surface_khr = create_vulkan_surface(vkinstance, window);
    const phys_dev = try get_physical_device(vkinstance, surface_khr, fba);
    _ = phys_dev;

    var renderer = try sdl.createRenderer(window, null, .{ .accelerated = true });
    defer {
        renderer.destroy();
    }

    event_loop: while (true) {
        if (sdl.pollEvent()) |event| {
            switch (event) {
                .key_down => |keydown| {
                    if (keydown.keycode == sdl.Keycode.escape) {
                        break :event_loop;
                    }
                },
                else => {},
            }
        }

        try renderer.setColor(sdl.Color.green);
        try renderer.clear();
        renderer.present();
    }
}

const atomic_package_counter_t = std.atomic.Value(u32);

fn create_vulkan_surface(vkinst: gfx.VkInstance, window: sdl.Window) gfx.VkSurfaceKHR {
    switch (builtin.os.tag) {
        .windows => {
            const wmi = window.getWMInfo() catch {
                @panic("Failed to get window data!");
            };

            std.log.info(
                "HINSTANCE 0x{x:0>8}, WIN 0x{x:0>8}",
                .{ @intFromPtr(wmi.u.win.hinstance), @intFromPtr(wmi.u.win.window) },
            );

            const surface_create_info = gfx.VkWin32SurfaceCreateInfoKHR{
                .sType = gfx.VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                .pNext = null,
                .flags = 0,
                .hinstance = @alignCast(@ptrCast(wmi.u.win.hinstance)),
                .hwnd = @alignCast(@ptrCast(wmi.u.win.window)),
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
