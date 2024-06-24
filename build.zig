const std = @import("std");
const sdl = @import("sdl");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addStaticLibrary(.{
        .name = "fractals",
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // This declares intent for the library to be installed into the standard
    // location when the user invokes the "install" step (the default step when
    // running `zig build`).
    b.installArtifact(lib);

    const exe = b.addExecutable(.{
        .name = "fractals",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const env_map = std.process.getEnvMap(b.allocator) catch @panic("Failed to get process environment map!");

    if (env_map.get("VK_SDK_PATH")) |path| {
        var tmp_buf: [512]u8 = undefined;
        exe.addLibraryPath(.{ .cwd_relative = std.fmt.bufPrint(&tmp_buf, "{s}/lib", .{path}) catch unreachable });

        var tmp_buf_incpath: [512]u8 = undefined;
        exe.addIncludePath(.{ .cwd_relative = std.fmt.bufPrint(&tmp_buf_incpath, "{s}/include", .{path}) catch unreachable });
    }

    const vk_lib_name = if (target.result.os.tag == .windows) "vulkan-1" else "vulkan";
    exe.linkSystemLibrary(vk_lib_name);

    // const vkzig_dep = b.dependency("vulkan_zig", .{
    //     .registry = @as([]const u8, b.pathFromRoot("src/third_party/vk/vk.xml")),
    // });
    // const vkzig_bindings = vkzig_dep.module("vulkan-zig");
    // exe.root_module.addImport("vulkan", vkzig_bindings);

    const sdl_sdk = sdl.init(b, null);
    sdl_sdk.link(exe, .static);
    exe.root_module.addImport("sdl2", sdl_sdk.getWrapperModule());

    const zigwin32 = b.dependency("zigwin32", .{});
    exe.root_module.addImport("zigwin32", zigwin32.module("zigwin32"));

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);

    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = b.addRunArtifact(exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
}
