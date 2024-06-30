{
  description = "An empty project that uses Zig.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
    zig.url = "github:mitchellh/zig-overlay";
    zls.url = "github:zigtools/zls";
    zls.inputs.nixpkgs.follows = "nixpkgs";
    gitignore.url = "github:hercules-ci/gitignore.nix";
    gitignore.inputs.nixpkgs.follows = "nixpkgs";

    # Used for shell.nix
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    gitignore,
    ...
  } @ inputs: let
    overlays = [
      # Other overlays
      (final: prev: {
        zigpkgs = inputs.zig.packages.${prev.system};
      })
    ];

    # Our supported systems are the same supported systems as the Zig binaries
    systems = builtins.attrNames inputs.zig.packages;
  in
    flake-utils.lib.eachSystem systems (
      system: let
        pkgs = import nixpkgs {
          inherit overlays system inputs;
          config.allowUnfree = true;
        };
      in rec {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            zigpkgs.master
            zig-shell-completions
            inputs.zls.packages.${pkgs.system}.zls

            gdb
            lldb
            seer
            gf
            renderdoc
            pkg-config
            vulkan-tools
            vulkan-headers
            vulkan-loader
            vulkan-tools-lunarg
            vulkan-validation-layers
            vulkan-utility-libraries
            vulkan-caps-viewer
            shaderc
            shaderc.bin
            shaderc.static
            shaderc.dev
            shaderc.lib
            glslang
            SDL2
            SDL2.dev
            SDL2_ttf
            SDL2_gfx
            bashInteractive

            vscode-extensions.vadimcn.vscode-lldb

            # libs
            zlib
            xorg.libX11
            xorg.libX11.dev
            xorg.libXi
            xorg.libXcursor
            xorg.libXrandr
            xorg.libXext
            xorg.libXinerama
            xorg.libXrender
            xorg.libXxf86vm
          ];

          shellHook = with pkgs; ''
            echo "Starting Zig development env ..."
              export LD_LIBRARY_PATH="";
              export PATH="$PATH:${lib.makeSearchPath "share/vscode/extensions/vadimcn.vscode-lldb/adapter" [vscode-extensions.vadimcn.vscode-lldb]}"
              export MONKA_GIGA="MONKA_MEGA";
          '';
        };

        # For compatibility with older versions of the `nix` binary
      }
    );
}
