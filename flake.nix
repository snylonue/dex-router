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
                pkgs.clang
                pkgs.foundry
                pkgs.graphviz
              ];
              buildInputs = [ pkgs.openblas ];

              LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
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
