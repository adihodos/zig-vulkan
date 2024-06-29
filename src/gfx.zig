const builtin = @import("builtin");

pub usingnamespace @cImport({
    if (builtin.os.tag == .windows) {
        @cDefine("VK_USE_PLATFORM_WIN32_KHR", {});
        @cDefine("WINVER", "0x0A00");
        @cDefine("_WIN32_WINNT", "0x0A00");
    }

    if (builtin.os.tag == .linux) {
        @cDefine("VK_USE_PLATFORM_XLIB_KHR", {});
    }
    @cInclude("vulkan/vulkan.h");
});
