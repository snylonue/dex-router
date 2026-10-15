{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs =
    inputs@{
      flake-parts,
      nixpkgs,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } (
      {
        ...
      }:
      {
        systems = [
          "x86_64-linux"
        ];
        perSystem =
          {
            system,
            pkgs,
            ...
          }:
          {
            _module.args.pkgs = import nixpkgs {
              inherit system;
              config.allowUnfree = true;
            };
            devShells.default = pkgs.mkShell {
              packages = [
                pkgs.cargo
                pkgs.rustc
                pkgs.rustfmt
                pkgs.clippy
                pkgs.rust-analyzer
                pkgs.foundry
                pkgs.graphviz
              ];
              # Keep Cargo, compiler and linker wrappers on the locked Nix toolchain.
              RUSTC = "${pkgs.rustc}/bin/rustc";
              RUSTDOC = "${pkgs.rustc}/bin/rustdoc";
              # Use the sccache installed on the host.
              RUSTC_WRAPPER = "sccache";
              RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (
                with pkgs;
                [
                  libGL
                  libxkbcommon
                  wayland
                ]
              );
            };
          };
      }
    );
}
