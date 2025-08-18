{
  description = "Development shell for HyperBEAM";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        erlang = pkgs.beam.interpreters.erlang_27;
        beamPackages = pkgs.beam.packagesWith erlang;
        rebar3 = beamPackages.rebar3;

        # Common build inputs
        commonInputs = with pkgs; [
          erlang
          rebar3
          rustc
          cargo
          cmake
          pkg-config
          git
          ncurses
          openssl
          nodejs_22
          python3
          lua
          gnumake
          curl
          cacert
          ninja
          gcc-unwrapped.lib
          gawk
        ];

        # Platform-specific inputs
        linuxInputs = with pkgs; [ rocksdb numactl ];
        darwinInputs = [ /* Add Darwin-specific if needed */ ];

      in {
        devShells.default = pkgs.mkShell {
          buildInputs = commonInputs
            ++ pkgs.lib.optionals pkgs.stdenv.isLinux linuxInputs
            ++ pkgs.lib.optionals pkgs.stdenv.isDarwin darwinInputs;

          shellHook = ''
              # Set platform-specific environment
              case "$OSTYPE" in
                linux*)
                    export CMAKE_LIBRARY_PATH="${pkgs.gcc-unwrapped.lib}/lib64:${pkgs.numactl}/lib:$CMAKE_LIBRARY_PATH"
                    preferred_shell=$(awk -F: -v user="$USER" '$1 == user {print $7}' /etc/passwd)
                    ;;
                darwin*)
                    export CMAKE_LIBRARY_PATH="${pkgs.gcc-unwrapped.lib}/lib:$CMAKE_LIBRARY_PATH"
                    preferred_shell=$(dscl . -read /Users/$USER UserShell 2>/dev/null | cut -d: -f2 | sed 's/^ //')
                    ;;
                esac
            if [ -z "$preferred_shell" ]; then
              preferred_shell="${pkgs.bashInteractive}/bin/bash"
            fi

            exec $preferred_shell -i
          '';
        };
      }
    );
}
