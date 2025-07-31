{
  description = "Pinkhaus Models - Shared database models and schemas";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        python = pkgs.python313;
        
        pinkhaus-models = python.pkgs.buildPythonPackage rec {
          pname = "pinkhaus-models";
          version = "0.1.0";
          format = "pyproject";
          
          src = ./.;
          
          nativeBuildInputs = with python.pkgs; [
            hatchling
          ];
          
          propagatedBuildInputs = with python.pkgs; [
            pydantic
            aiosqlite
          ];
          
          # Disable tests for now
          doCheck = false;
          
          meta = with pkgs.lib; {
            description = "Shared database models and schemas for pinkhaus projects";
            homepage = "https://github.com/altfund/pinkhaus";
            license = licenses.mit;
          };
        };
      in
      {
        packages = {
          default = pinkhaus-models;
          pinkhaus-models = pinkhaus-models;
        };
        
        # For development
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python
            python.pkgs.pip
            python.pkgs.hatchling
            python.pkgs.pydantic
            python.pkgs.aiosqlite
          ];
        };
      });
}