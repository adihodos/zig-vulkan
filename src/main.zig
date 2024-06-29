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

const AllocatorBundle = struct {
    gpa: *std.heap.GeneralPurposeAllocator,
    fixed: *std.heap.FixedBufferAllocator,
};

pub fn main() !void {
    var general_alloc = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        _ = general_alloc.deinit();
    }

    var fixed_allocator = std.heap.FixedBufferAllocator.init(&g_scratch_buffer);

    const alloc_bundle = AllocatorBundle{
        .gpa = &general_alloc,
        .fixed = &fixed_allocator,
    };

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

    const vulkan_renderer = try VulkanRenderer.init(window, alloc_bundle);
    defer {
        vulkan_renderer.deinit();
    }

    var renderer = try sdl.createRenderer(window, null, .{ .accelerated = true });
    defer {
        renderer.destroy();
    }

    const wmi = window.getWMInfo() catch {
        @panic("Failed to get window data!");
    };
    std.log.info("X11 display 0x{x:8>}, window {d}", .{ @intFromPtr(wmi.u.x11.display), wmi.u.x11.window });

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

fn create_vulkan_instance(fba: std.mem.Allocator) !gfx.VkInstance {
    {
        //
        // print available extensions
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
    }

    {
        //
        // print available layers
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
    }

    const enabled_layers = [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"};
    const enabled_extensions = [_][*:0]const u8{
        "VK_KHR_surface",
        switch (builtin.os.tag) {
            .windows => "VK_KHR_win32_surface",
            .linux => "VK_KHR_xlib_surface",
            else => @panic("Platform not supported yet!"),
        },
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
        .pApplicationName = "zig_vk_app",
        .applicationVersion = gfx.VK_MAKE_API_VERSION(1, 0, 0, 0),
        .pEngineName = "zig_vk_engine",
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
        return error.VulkanApiError;
    }

    std.log.info("Vulkan instance created @ 0x{x:8>}", .{@intFromPtr(instance)});
    return instance;
}

fn create_vulkan_surface(vkinst: gfx.VkInstance, window: sdl.Window) !gfx.VkSurfaceKHR {
    const wmi = try window.getWMInfo();

    const surface_handle, const result = make_surface: {
        switch (builtin.os.tag) {
            .windows => {
                std.log.info("HINSTANCE 0x{x:0>8}, WIN 0x{x:0>8}", .{ @intFromPtr(wmi.u.win.hinstance), @intFromPtr(wmi.u.win.window) });

                const surface_create_info = gfx.VkWin32SurfaceCreateInfoKHR{
                    .sType = gfx.VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                    .pNext = null,
                    .flags = 0,
                    .hinstance = @alignCast(@ptrCast(wmi.u.win.hinstance)),
                    .hwnd = @alignCast(@ptrCast(wmi.u.win.window)),
                };

                var surface: gfx.VkSurfaceKHR = null;
                const result = gfx.vkCreateWin32SurfaceKHR(vkinst, &surface_create_info, null, &surface);
                break :make_surface .{ surface, result };
            },
            .linux => {
                std.log.info(
                    "X11 display 0x{x:8>}, window {d}",
                    .{ @intFromPtr(wmi.u.x11.display), wmi.u.x11.window },
                );
                const xlib_surface_create_info = gfx.VkXlibSurfaceCreateInfoKHR{
                    .sType = gfx.VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
                    .pNext = null,
                    .flags = 0,
                    .dpy = @ptrCast(wmi.u.x11.display),
                    .window = wmi.u.x11.window,
                };
                var xlib_surface: gfx.VkSurfaceKHR = null;
                const result = gfx.vkCreateXlibSurfaceKHR(vkinst, &xlib_surface_create_info, null, &xlib_surface);
                break :make_surface .{ xlib_surface, result };
            },
            else => @panic("Not implemented for this platform!"),
        }
    };

    if (result != gfx.VK_SUCCESS or surface_handle == null) {
        std.log.err("Failed to create VkSurfaceKHR, error {d}", .{result});
        return error.VulkanApiError;
    }

    return surface_handle;
}

const GraphicsSystemError = error{
    NoSuitableDeviceFound,
    VulkanApiError,
};

const PhysicalDeviceState = struct {
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

    pub fn create(device: gfx.VkPhysicalDevice) PhysicalDeviceState {
        var pdd: PhysicalDeviceState = PhysicalDeviceState{
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

    pub fn check_feature_support(self: *const PhysicalDeviceState) bool {
        return self.features_vk11.shaderDrawParameters != 0 and
            self.features_vk12.descriptorIndexing != 0 and
            self.features_vk12.descriptorBindingPartiallyBound != 0;
    }
};

const PhysicalDeviceWithSurfaceData = struct {
    pdd: PhysicalDeviceState,
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
        const pdd = PhysicalDeviceState.create(pd);
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
            std.log.info(
                "Rejecting device {s}, does not support all required features.",
                .{pdd.props.properties.deviceName},
            );
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

const LogicalDeviceState = struct {
    device: gfx.VkDevice,
    queue_family: u32,
    queue_idx: u32,
    queue: gfx.VkQueue,

    pub fn deinit(self: *const LogicalDeviceState) void {
        gfx.vkDestroyDevice(self.device, null);
    }
};

fn create_logical_device(ds: *const PhysicalDeviceWithSurfaceData) !LogicalDeviceState {
    const queue_priorities = [_]f32{1.0};

    const queue_create_info = [_]gfx.VkDeviceQueueCreateInfo{
        gfx.VkDeviceQueueCreateInfo{
            .sType = gfx.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueFamilyIndex = ds.queue_family,
            .queueCount = @intCast(queue_priorities.len),
            .pQueuePriorities = &queue_priorities,
        },
    };

    const device_exts = [_][*:0]const u8{
        gfx.VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        gfx.VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
        gfx.VK_KHR_MAINTENANCE1_EXTENSION_NAME,
    };

    var f2 = ds.pdd.features;
    var f11 = ds.pdd.features_vk11;
    var f12 = ds.pdd.features_vk12;
    var f13 = ds.pdd.features_vk13;

    f2.pNext = &f11;
    f11.pNext = &f12;
    f12.pNext = &f13;
    f13.pNext = null;

    const device_create_info = gfx.VkDeviceCreateInfo{
        .sType = gfx.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &f2,
        .flags = 0,
        .queueCreateInfoCount = @intCast(queue_create_info.len),
        .pQueueCreateInfos = &queue_create_info,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = null,
        .enabledExtensionCount = @intCast(device_exts.len),
        .ppEnabledExtensionNames = &device_exts,
        .pEnabledFeatures = null,
    };

    var logical_device: gfx.VkDevice = null;
    const result = gfx.vkCreateDevice(ds.pdd.device, &device_create_info, null, &logical_device);
    if (result != gfx.VK_SUCCESS or logical_device == null) {
        std.log.err("Failed to create logical device, error {d}", .{result});
        return error.VulkanApiError;
    }

    var queue: gfx.VkQueue = null;
    gfx.vkGetDeviceQueue(logical_device, ds.queue_family, 0, &queue);

    return LogicalDeviceState{
        .device = logical_device,
        .queue = queue,
        .queue_family = ds.queue_family,
        .queue_idx = 0,
    };
}

const ComponentMappingIdentity = gfx.VkComponentMapping{
    .r = gfx.VK_COMPONENT_SWIZZLE_IDENTITY,
    .g = gfx.VK_COMPONENT_SWIZZLE_IDENTITY,
    .b = gfx.VK_COMPONENT_SWIZZLE_IDENTITY,
    .a = gfx.VK_COMPONENT_SWIZZLE_IDENTITY,
};

const UniqueImageWithMemory = struct {
    image: gfx.VkImage,
    memory: gfx.VkDeviceMemory,

    pub fn init(
        device: gfx.VkDevice,
        image_create_info: *const gfx.VkImageCreateInfo,
        mem_props: *const gfx.VkPhysicalDeviceMemoryProperties,
        mem_type: gfx.VkMemoryPropertyFlags,
    ) !UniqueImageWithMemory {
        var image_ptr: gfx.VkImage = null;
        const create_img_res = vulkan_api_call(gfx.vkCreateImage, .{ device, image_create_info, null, &image_ptr });
        if (create_img_res != gfx.VK_SUCCESS or image_ptr == null) {
            return error.VulkanApiError;
        }

        errdefer {
            gfx.vkDestroyImage(device, image_ptr, null);
        }

        return UniqueImageWithMemory.init_with_image(device, image_ptr, mem_props, mem_type);
    }

    pub fn init_with_image(
        device: gfx.VkDevice,
        image: gfx.VkImage,
        mem_props: *const gfx.VkPhysicalDeviceMemoryProperties,
        memory_type: gfx.VkMemoryPropertyFlags,
    ) !UniqueImageWithMemory {
        var memory_req: gfx.VkMemoryRequirements = undefined;
        vulkan_api_call(gfx.vkGetImageMemoryRequirements, .{ device, image, &memory_req });

        const memory_alloc_info = make_vulkan_struct(
            gfx.VkMemoryAllocateInfo,
            .{
                .allocationSize = memory_req.size,
                .memoryTypeIndex = find_memory_type(mem_props, memory_type).?,
            },
        );

        var image_memory: gfx.VkDeviceMemory = null;
        const alloc_result = vulkan_api_call(gfx.vkAllocateMemory, .{ device, &memory_alloc_info, null, &image_memory });
        if (alloc_result != gfx.VK_SUCCESS) {
            return error.VulkanApiError;
        }

        errdefer {
            gfx.vkFreeMemory(device, image_memory, null);
        }

        const bind_result = vulkan_api_call(gfx.vkBindImageMemory, .{ device, image, image_memory, 0 });
        if (bind_result != gfx.VK_SUCCESS)
            return error.VulkanApiError;

        return UniqueImageWithMemory{ .image = image, .memory = image_memory };
    }

    pub fn deinit(self: *const UniqueImageWithMemory, device: gfx.VkDevice) void {
        vulkan_api_call(gfx.vkFreeMemory, .{ device, self.memory, null });
        vulkan_api_call(gfx.vkDestroyImage, .{ device, self.image, null });
    }
};

const UniqueImageWithView = struct {
    image: UniqueImageWithMemory,
    view: gfx.VkImageView,

    pub fn init(
        image: UniqueImageWithMemory,
        device: gfx.VkDevice,
        image_view_create_info: *const gfx.VkImageViewCreateInfo,
    ) !UniqueImageWithView {
        var image_view: gfx.VkImageView = null;
        const image_view_create_result = vulkan_api_call(gfx.vkCreateImageView, .{ device, image_view_create_info, null, &image_view });
        if (image_view_create_result != gfx.VK_SUCCESS or image_view == null) {
            return error.VulkanApiError;
        }

        return UniqueImageWithView{
            .image = image,
            .view = image_view,
        };
    }

    pub fn deinit(self: *const @This(), device: gfx.VkDevice) void {
        gfx.vkDestroyImageView(device, self.view, null);
        self.image.deinit(device);
    }
};

const FrameState = struct {
    swapchain_image: gfx.VkImage,
    swapchain_view: gfx.VkImageView,
    depth_stencil: UniqueImageWithView,
    fence: gfx.VkFence,
    semaphore_present: gfx.VkSemaphore,
    semaphore_render: gfx.VkSemaphore,

    pub fn deinit(self: *const FrameState, device: gfx.VkDevice) void {
        self.depth_stencil.deinit(device);
        gfx.vkDestroyImageView(device, self.swapchain_view, null);
        gfx.vkDestroyFence(device, self.fence, null);
        gfx.vkDestroySemaphore(device, self.semaphore_present, null);
        gfx.vkDestroySemaphore(device, self.semaphore_render, null);
    }

    pub fn init(device: gfx.VkDevice, swapchain_image: gfx.VkImage, dps: *const PhysicalDeviceWithSurfaceData) !FrameState {
        const swapchain_view = create_swapchain_image_view: {
            const image_create_info = gfx.VkImageViewCreateInfo{
                .sType = gfx.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = null,
                .flags = 0,
                .image = swapchain_image,
                .viewType = gfx.VK_IMAGE_VIEW_TYPE_2D,
                .format = dps.surface_format.format,
                .components = ComponentMappingIdentity,
                .subresourceRange = gfx.VkImageSubresourceRange{
                    .aspectMask = gfx.VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };

            var image_view: gfx.VkImageView = null;
            const result = gfx.vkCreateImageView(device, &image_create_info, null, &image_view);
            if (result != gfx.VK_SUCCESS or image_view == null) {
                std.log.err("Failed to create image view, error {d}", .{result});
                return error.VulkanApiError;
            }

            break :create_swapchain_image_view image_view;
        };
        errdefer {
            gfx.vkDestroyImageView(device, swapchain_view, null);
        }

        const depth_stencil_state = create_depth_stencil_state: {
            const image_create_info = make_vulkan_struct(gfx.VkImageCreateInfo, .{
                .imageType = gfx.VK_IMAGE_TYPE_2D,
                // TODO: depth stencil format when creating the logical device !!!!
                .format = gfx.VK_FORMAT_D32_SFLOAT_S8_UINT,
                .extent = gfx.VkExtent3D{
                    .width = dps.surface_caps.currentExtent.width,
                    .height = dps.surface_caps.currentExtent.height,
                    .depth = 1,
                },
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = gfx.VK_SAMPLE_COUNT_1_BIT,
                .tiling = gfx.VK_IMAGE_TILING_OPTIMAL,
                .usage = gfx.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                .sharingMode = gfx.VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 1,
                .pQueueFamilyIndices = &dps.queue_family,
                .initialLayout = gfx.VK_IMAGE_LAYOUT_UNDEFINED,
            });

            const depth_stencil_image = try UniqueImageWithMemory.init(
                device,
                &image_create_info,
                &dps.pdd.memory,
                gfx.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            );
            errdefer {
                depth_stencil_image.deinit(device);
            }

            const image_view_create_info = make_vulkan_struct(gfx.VkImageViewCreateInfo, .{
                .image = depth_stencil_image.image,
                .viewType = gfx.VK_IMAGE_VIEW_TYPE_2D,
                .format = gfx.VK_FORMAT_D32_SFLOAT_S8_UINT,
                .components = ComponentMappingIdentity,
                .subresourceRange = gfx.VkImageSubresourceRange{
                    .aspectMask = gfx.VK_IMAGE_ASPECT_DEPTH_BIT | gfx.VK_IMAGE_ASPECT_STENCIL_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            });

            break :create_depth_stencil_state try UniqueImageWithView.init(depth_stencil_image, device, &image_view_create_info);
        };
        errdefer {
            depth_stencil_state.deinit(device);
        }

        const fence = create_fence: {
            var fence: gfx.VkFence = null;
            const fence_create_info = make_vulkan_struct(gfx.VkFenceCreateInfo, .{ .flags = gfx.VK_FENCE_CREATE_SIGNALED_BIT });

            const result = gfx.vkCreateFence(device, &fence_create_info, null, &fence);
            if (result != gfx.VK_SUCCESS) {
                std.log.err("Failed to create fence, error {d}", .{result});
                return error.VulkanApiError;
            }

            break :create_fence fence;
        };

        errdefer {
            gfx.vkDestroyFence(device, fence, null);
        }

        var semaphores = [_]gfx.VkSemaphore{ null, null };
        for (&semaphores) |*sem| {
            var s: gfx.VkSemaphore = null;
            const semaphore_create_info = make_vulkan_struct(gfx.VkSemaphoreCreateInfo, .{});
            const result = gfx.vkCreateSemaphore(device, &semaphore_create_info, null, &s);

            if (result != gfx.VK_SUCCESS) {
                std.log.err("Failed to create semaphore, error {d}", .{result});
                return error.VulkanApiError;
            }

            sem.* = s;
        }

        return FrameState{
            .swapchain_image = swapchain_image,
            .swapchain_view = swapchain_view,
            .depth_stencil = depth_stencil_state,
            .fence = fence,
            .semaphore_present = semaphores[0],
            .semaphore_render = semaphores[1],
        };
    }
};

const SwapchainState = struct {
    handle: gfx.VkSwapchainKHR,
    image_count: u32,
    frame_index: u32,
    frame_states: std.ArrayList(FrameState),

    fn deinit(self: *const SwapchainState, device: gfx.VkDevice) void {
        for (self.frame_states.items) |fs| {
            fs.deinit(device);
        }
        self.frame_states.deinit();
        gfx.vkDestroySwapchainKHR(device, self.handle, null);
    }

    fn init(
        ldev: *const LogicalDeviceState,
        prev_swapchain: ?gfx.VkSwapchainKHR,
        dps: *const PhysicalDeviceWithSurfaceData,
        allocator: std.mem.Allocator,
    ) !SwapchainState {
        const image_count = std.math.clamp(dps.surface_caps.minImageCount + 2, dps.surface_caps.minImageCount, dps.surface_caps.maxImageCount);
        if (dps.surface_caps.currentExtent.width == 0xffffffff or dps.surface_caps.currentExtent.height == 0xffffffff) {
            //
            //
        }

        const queue_families = [_]u32{dps.queue_family};
        const swapchain_create_info = gfx.VkSwapchainCreateInfoKHR{
            .sType = gfx.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .pNext = null,
            .flags = 0,
            .surface = dps.surface,
            .minImageCount = image_count,
            .imageFormat = dps.surface_format.format,
            .imageColorSpace = dps.surface_format.colorSpace,
            .imageExtent = dps.surface_caps.currentExtent,
            .imageArrayLayers = 1,
            .imageUsage = gfx.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = gfx.VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = @intCast(queue_families.len),
            .pQueueFamilyIndices = &queue_families,
            .preTransform = gfx.VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
            .compositeAlpha = gfx.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = dps.present_mode,
            .clipped = gfx.VK_FALSE,
            .oldSwapchain = prev_swapchain orelse null,
        };

        const swapchain = create_swapchain: {
            var swapchain: gfx.VkSwapchainKHR = null;
            const result = gfx.vkCreateSwapchainKHR(ldev.device, &swapchain_create_info, null, &swapchain);
            if (result != gfx.VK_SUCCESS) {
                std.log.err("Failed to create swapchain, error {d}", .{result});
                return error.VulkanApiError;
            }

            break :create_swapchain swapchain;
        };

        errdefer {
            gfx.vkDestroySwapchainKHR(ldev.device, swapchain, null);
        }

        const swapchain_images = get_swapchain_images: {
            var img_count: u32 = 0;
            const result = vulkan_api_call(gfx.vkGetSwapchainImagesKHR, .{ ldev.device, swapchain, &img_count, null });
            if (result != gfx.VK_SUCCESS or img_count == 0) {
                return error.VulkanApiError;
            }

            var swapchain_images = try std.ArrayList(gfx.VkImage).initCapacity(allocator, img_count);
            errdefer {
                swapchain_images.deinit();
            }
            try swapchain_images.resize(img_count);

            const fill_result = vulkan_api_call(
                gfx.vkGetSwapchainImagesKHR,
                .{ ldev.device, swapchain, &img_count, swapchain_images.items.ptr },
            );
            if (fill_result != gfx.VK_SUCCESS) {
                return error.VulkanApiError;
            }

            break :get_swapchain_images swapchain_images;
        };

        defer {
            swapchain_images.deinit();
        }

        var frame_state = try std.ArrayList(FrameState).initCapacity(allocator, swapchain_images.items.len);
        for (swapchain_images.items) |swapchain_image| {
            try frame_state.append(try FrameState.init(ldev.device, swapchain_image, dps));
        }
        errdefer {
            frame_state.deinit();
        }

        return SwapchainState{
            .handle = swapchain,
            .image_count = @intCast(swapchain_images.items.len),
            .frame_index = 0,
            .frame_states = frame_state,
        };
    }
};

const SurfaceKHRState = struct {
    surface: gfx.VkSurfaceKHR,
    surface_format: gfx.VkSurfaceFormatKHR,
    depth_stencil_format: gfx.VkFormat,
    surface_caps: gfx.VkSurfaceCapabilitiesKHR,
    present_mode: gfx.VkPresentModeKHR,
};

const VulkanDynamicDispatch = struct {
    vkCreateDebugUtilsMessengerEXT: @typeInfo(gfx.PFN_vkCreateDebugUtilsMessengerEXT).Optional.child,
    vkDestroyDebugUtilsMessengerEXT: @typeInfo(gfx.PFN_vkDestroyDebugUtilsMessengerEXT).Optional.child,

    pub fn init(instance: gfx.VkInstance) !VulkanDynamicDispatch {
        var dispatch: VulkanDynamicDispatch = undefined;

        inline for (@typeInfo(@This()).Struct.fields) |field| {
            std.log.info("Field ptr {s}, {s}", .{ field.name, @typeName(field.type) });
            const func_ptr = gfx.vkGetInstanceProcAddr(instance, field.name);
            if (func_ptr == null) {
                std.log.err("Failed to load function pointer {s}", .{field.name});
                return error.VulkanApiError;
            }
            @field(dispatch, field.name) = @ptrCast(func_ptr);
            std.log.info("Loaded function pointer {s} @ 0x{x:8>0}", .{ field.name, @intFromPtr(@field(dispatch, field.name)) });
        }

        return dispatch;
    }
};

const VulkanDebugState = struct {
    debug_utils_msgr: gfx.VkDebugUtilsMessengerEXT,

    pub fn init(instance: gfx.VkInstance, dispatch: *const VulkanDynamicDispatch) !VulkanDebugState {
        const dbg_utils_create_info = make_vulkan_struct(gfx.VkDebugUtilsMessengerCreateInfoEXT, .{
            .messageSeverity = gfx.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | gfx.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
            .messageType = gfx.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | gfx.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | gfx.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = vk_dbg_utils_msg_callback,
            .pUserData = null,
        });

        var debug_utils_msgr: gfx.VkDebugUtilsMessengerEXT = null;
        const result = vulkan_api_call(dispatch.vkCreateDebugUtilsMessengerEXT, .{ instance, &dbg_utils_create_info, null, &debug_utils_msgr });
        if (result != gfx.VK_SUCCESS)
            return error.VulkanApiError;

        return VulkanDebugState{ .debug_utils_msgr = debug_utils_msgr };
    }

    fn deinit(
        self: *const VulkanDebugState,
        instance: gfx.VkInstance,
        dispatch: *const VulkanDynamicDispatch,
    ) void {
        vulkan_api_call(dispatch.vkDestroyDebugUtilsMessengerEXT, .{ instance, self.debug_utils_msgr, null });
    }
};

const VulkanRenderer = struct {
    instance: gfx.VkInstance,
    dyn_dispatch: VulkanDynamicDispatch,
    dbg: VulkanDebugState,
    physical: PhysicalDeviceState,
    logical: LogicalDeviceState,
    surface: SurfaceKHRState,
    swapchain: SwapchainState,

    pub fn init(window: sdl.Window, alloc: AllocatorBundle) !VulkanRenderer {
        const instance = try create_vulkan_instance(alloc.fixed.allocator());
        const dyn_dispatch = try VulkanDynamicDispatch.init(instance);

        const debug_state = try VulkanDebugState.init(instance, &dyn_dispatch);
        const surface_khr = try create_vulkan_surface(instance, window);
        std.log.info("Create Vulkan surface (VkSurfaceKHR @ 0x{x:8>})", .{@intFromPtr(surface_khr)});

        const phys_dev = try get_physical_device(instance, surface_khr, alloc.fixed.allocator());
        const device = try create_logical_device(&phys_dev);

        std.log.info(
            "Created logical device @ 0x{x:8>}, queue @ 0x{x:8>}",
            .{ @intFromPtr(device.device), @intFromPtr(device.queue) },
        );

        const swapchain_state = try SwapchainState.init(&device, null, &phys_dev, alloc.gpa.*);

        return VulkanRenderer{
            .instance = instance,
            .dyn_dispatch = dyn_dispatch,
            .dbg = debug_state,
            .physical = phys_dev.pdd,
            .logical = device,
            .surface = SurfaceKHRState{
                .surface = surface_khr,
                .surface_format = phys_dev.surface_format,
                .depth_stencil_format = gfx.VK_FORMAT_D32_SFLOAT_S8_UINT,
                .surface_caps = phys_dev.surface_caps,
                .present_mode = phys_dev.present_mode,
            },
            .swapchain = swapchain_state,
        };
    }

    pub fn deinit(self: *const VulkanRenderer) void {
        _ = gfx.vkQueueWaitIdle(self.logical.queue);
        self.swapchain.deinit(self.logical.device);
        gfx.vkDestroySurfaceKHR(self.instance, self.surface.surface, null);
        self.logical.deinit();
        self.dbg.deinit(self.instance, &self.dyn_dispatch);
        gfx.vkDestroyInstance(self.instance, null);
    }
};

fn vulkan_api_call(vkfunc: anytype, func_args: anytype) switch (@typeInfo(@TypeOf(vkfunc))) {
    .Pointer => |fnptr| @typeInfo(fnptr.child).Fn.return_type.?,
    .Fn => |func| func.return_type.?,

    else => {
        @compileError("Unsupported");
    },
} {
    const arg_pack_type = @typeInfo(@TypeOf(func_args));
    if (arg_pack_type != .Struct) {
        @compileError("expected tuple or struct argument, found " ++ @typeName(@TypeOf(func_args)));
    }

    const return_type = switch (@typeInfo(@TypeOf(vkfunc))) {
        .Pointer => |fnptr| @typeInfo(fnptr.child).Fn.return_type.?,
        .Fn => |func| func.return_type.?,
        else => {
            @compileError("Unsupported");
        },
    };
    const func_params = switch (@typeInfo(@TypeOf(vkfunc))) {
        .Pointer => |fnptr| @typeInfo(fnptr.child).Fn.params,
        .Fn => |func| func.params,
        else => {
            @compileError("Unsupported");
        },
    };

    if (arg_pack_type.Struct.fields.len != func_params.len)
        @compileError("Wrong number of arguments to call of " ++ @typeName(@TypeOf(vkfunc)));

    switch (@typeInfo(return_type)) {
        .Void => {
            @call(std.builtin.CallModifier.auto, vkfunc, func_args);
        },
        .Int => {
            const return_value = @call(std.builtin.CallModifier.auto, vkfunc, func_args);
            if (return_value != gfx.VK_SUCCESS) {
                std.log.err("Vulkan API error: {d} - 0x{x:8>0}", .{ return_value, return_value });
            }
            return return_value;
        },
        else => {
            @compileError("This function's return type is not supported yet!");
        },
    }
}

fn make_vulkan_struct(comptime T: type, args: anytype) T {
    const type_hash = comptime std.hash.Fnv1a_64.hash(@typeName(T));
    var result: T = undefined;

    if (!@hasField(T, "sType"))
        @compileError("Type " ++ @typeName(T) ++ " is missing a required field (sType). Is this a Vulkan API structure ?!");

    if (@hasField(T, "pNext"))
        @field(result, "pNext") = null;

    if (@hasField(T, "flags"))
        @field(result, "flags") = 0;

    switch (type_hash) {
        std.hash.Fnv1a_64.hash(@typeName(gfx.VkImageCreateInfo)) => {
            result.sType = gfx.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        },
        std.hash.Fnv1a_64.hash(@typeName(gfx.VkImageViewCreateInfo)) => {
            result.sType = gfx.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        },
        std.hash.Fnv1a_64.hash(@typeName(gfx.VkSemaphoreCreateInfo)) => {
            result.sType = gfx.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        },
        std.hash.Fnv1a_64.hash(@typeName(gfx.VkFenceCreateInfo)) => {
            result.sType = gfx.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        },
        std.hash.Fnv1a_64.hash(@typeName(gfx.VkMemoryAllocateInfo)) => {
            result.sType = gfx.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        },
        std.hash.Fnv1a_64.hash(@typeName(gfx.VkDebugUtilsMessengerCreateInfoEXT)) => {
            result.sType = gfx.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        },
        else => @compileError("Type " ++ @typeName(T) ++ " not supported!"),
    }

    inline for (@typeInfo(@TypeOf(args)).Struct.fields) |args_field| {
        if (!@hasField(T, args_field.name))
            @compileError("Type " ++ @typeName(T) ++ " does not have field " ++ args_field.name);

        @field(result, args_field.name) = @field(args, args_field.name);
    }

    return result;
}

fn find_memory_type(mem_props: *const gfx.VkPhysicalDeviceMemoryProperties, req_props: gfx.VkMemoryPropertyFlags) ?u32 {
    var memory_index: u32 = 0;
    while (memory_index < mem_props.memoryTypeCount) : (memory_index += 1) {
        const properties = mem_props.memoryTypes[memory_index].propertyFlags;

        if ((properties & req_props) != 0) {
            return memory_index;
        }
    }

    return null;
}
