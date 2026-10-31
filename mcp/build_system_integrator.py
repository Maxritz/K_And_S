"""
Build System Integrator for K_And_S MCP Server

Integrates with various build systems (CMake, Make, Meson, Bazel, etc.)
"""

from __future__ import annotations
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

# Import enhanced components
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@dataclass
class BuildIntegrationResult:
    """Result of build system integration"""
    status: str
    build_system: str
    files_created: List[str]
    files_modified: List[str]
    backup_files: List[str]
    errors: List[str]
    warnings: List[str]

class BuildSystemIntegrator:
    """Integrates with various build systems"""
    
    def __init__(self):
        self.build_systems = {
            "cmake": self._integrate_cmake,
            "make": self._integrate_make,
            "meson": self._integrate_meson,
            "bazel": self._integrate_bazel,
            "ninja": self._integrate_ninja,
            "scons": self._integrate_scons
        }
    
    async def integrate_build_system(self, project_path: str, build_system: str, backend_config: Dict[str, Any]) -> BuildIntegrationResult:
        """Integrate with specified build system"""
        if build_system not in self.build_systems:
            return BuildIntegrationResult(
                status="failed",
                build_system=build_system,
                files_created=[],
                files_modified=[],
                backup_files=[],
                errors=[f"Unsupported build system: {build_system}"],
                warnings=[]
            )
        
        try:
            result = await self.build_systems[build_system](project_path, backend_config)
            return result
        except Exception as e:
            return BuildIntegrationResult(
                status="failed",
                build_system=build_system,
                files_created=[],
                files_modified=[],
                backup_files=[],
                errors=[str(e)],
                warnings=[]
            )
    
    async def _integrate_cmake(self, project_path: str, backend_config: Dict[str, Any]) -> BuildIntegrationResult:
        """Integrate with CMake build system"""
        cmake_template = self._generate_cmake_template(backend_config)
        cmake_file = Path(project_path) \ "CMakeLists.txt"
        
        files_created = []
        files_modified = []
        backup_files = []
        
        # Backup existing CMakeLists.txt
        if cmake_file.exists():
            backup_file = cmake_file.with_suffix('.txt.bak')
            cmake_file.rename(backup_file)
            backup_files.append(str(backup_file))
        
        # Generate new CMakeLists.txt
        cmake_file.write_text(cmake_template, encoding="utf-8")
        files_created.append(str(cmake_file))
        
        return BuildIntegrationResult(
            status="success",
            build_system="cmake",
            files_created=files_created,
            files_modified=files_modified,
            backup_files=backup_files,
            errors=[],
            warnings=[]
        )
    
    async def _integrate_make(self, project_path: str, backend_config: Dict[str, Any]) -> BuildIntegrationResult:
        """Integrate with Make build system"""
        makefile_template = self._generate_makefile_template(backend_config)
        makefile = Path(project_path) \ "Makefile"
        
        files_created = []
        backup_files = []
        
        if makefile.exists():
            backup_file = makefile.with_suffix('.bak')
            makefile.rename(backup_file)
            backup_files.append(str(backup_file))
        
        makefile.write_text(makefile_template, encoding="utf-8")
        files_created.append(str(makefile))
        
        return BuildIntegrationResult(
            status="success",
            build_system="make",
            files_created=files_created,
            files_modified=[],
            backup_files=backup_files,
            errors=[],
            warnings=[]
        )
    
    async def _integrate_meson(self, project_path: str, backend_config: Dict[str, Any]) -> BuildIntegrationResult:
        """Integrate with Meson build system"""
        meson_template = self._generate_meson_template(backend_config)
        meson_file = Path(project_path) \ "meson.build"
        
        files_created = []
        backup_files = []
        
        if meson_file.exists():
            backup_file = meson_file.with_suffix('.bak')
            meson_file.rename(backup_file)
            backup_files.append(str(backup_file))
        
        meson_file.write_text(meson_template, encoding="utf-8")
        files_created.append(str(meson_file))
        
        return BuildIntegrationResult(
            status="success",
            build_system="meson",
            files_created=files_created,
            files_modified=[],
            backup_files=backup_files,
            errors=[],
            warnings=[]
        )
    
    async def _integrate_bazel(self, project_path: str, backend_config: Dict[str, Any]) -> BuildIntegrationResult:
        """Integrate with Bazel build system"""
        bazel_template = self._generate_bazel_template(backend_config)
        bazel_file = Path(project_path) \ "BUILD"
        
        files_created = []
        backup_files = []
        
        if bazel_file.exists():
            backup_file = bazel_file.with_suffix('.bak')
            bazel_file.rename(backup_file)
            backup_files.append(str(backup_file))
        
        bazel_file.write_text(bazel_template, encoding="utf-8")
        files_created.append(str(bazel_file))
        
        return BuildIntegrationResult(
            status="success",
            build_system="bazel",
            files_created=files_created,
            files_modified=[],
            backup_files=backup_files,
            errors=[],
            warnings=[]
        )
    
    async def _integrate_ninja(self, project_path: str, backend_config: Dict[str, Any]) -> BuildIntegrationResult:
        """Integrate with Ninja build system"""
        ninja_template = self._generate_ninja_template(backend_config)
        ninja_file = Path(project_path) \ "build.ninja"
        
        files_created = []
        backup_files = []
        
        if ninja_file.exists():
            backup_file = ninja_file.with_suffix('.bak')
            ninja_file.rename(backup_file)
            backup_files.append(str(backup_file))
        
        ninja_file.write_text(ninja_template, encoding="utf-8")
        files_created.append(str(ninja_file))
        
        return BuildIntegrationResult(
            status="success",
            build_system="ninja",
            files_created=files_created,
            files_modified=[],
            backup_files=backup_files,
            errors=[],
            warnings=[]
        )
    
    async def _integrate_scons(self, project_path: str, backend_config: Dict[str, Any]) -> BuildIntegrationResult:
        """Integrate with SCons build system"""
        scons_template = self._generate_scons_template(backend_config)
        scons_file = Path(project_path) \ "SConstruct"
        
        files_created = []
        backup_files = []
        
        if scons_file.exists():
            backup_file = scons_file.with_suffix('.bak')
            scons_file.rename(backup_file)
            backup_files.append(str(backup_file))
        
        scons_file.write_text(scons_template, encoding="utf-8")
        files_created.append(str(scons_file))
        
        return BuildIntegrationResult(
            status="success",
            build_system="scons",
            files_created=files_created,
            files_modified=[],
            backup_files=backup_files,
            errors=[],
            warnings=[]
        )
    
    def _generate_cmake_template(self, backend_config: Dict[str, Any]) -> str:
        """Generate CMake template"""
        compiler = backend_config.get("compiler", "gcc")
        extensions = backend_config.get("extensions", [".cpp"])
        
        return f"""# CMakeLists.txt - Generated by K_And_S Build System Integrator
cmake_minimum_required(VERSION 3.10)
project(KernelPorting LANGUAGES CXX)

# Backend configuration
set(BACKEND "{backend_config.get('backend', 'cpu')}")
set(COMPILER "{compiler}")

# Find required packages
find_package(Threads REQUIRED)

# Source files
file(GLOB SOURCES "src\*.{extensions[0].lstrip('.')}")

# Create executable
add_executable(kernel_port ${{SOURCES}})

# Link libraries
target_link_libraries(kernel_port Threads::Threads)

# Compiler flags
target_compile_options(kernel_port PRIVATE -O3 -std=c++17)

# Backend-specific flags
if(BACKEND STREQUAL "hip")
    target_compile_options(kernel_port PRIVATE --offload-arch=gfx1201)
elseif(BACKEND STREQUAL "cuda")
    target_compile_options(kernel_port PRIVATE -arch=sm_80)
endif()

# Installation
install(TARGETS kernel_port DESTINATION bin)
"""
    
    def _generate_makefile_template(self, backend_config: Dict[str, Any]) -> str:
        """Generate Makefile template"""
        compiler = backend_config.get("compiler", "gcc")
        extensions = backend_config.get("extensions", [".cpp"])
        
        return f"""# Makefile - Generated by K_And_S Build System Integrator
CC = {compiler}
CFLAGS = -O3 -std=c++17
BACKEND = {backend_config.get('backend', 'cpu')}

# Source files
SOURCES = $(wildcard src\*{extensions[0]})
OBJECTS = $(SOURCES:.{extensions[0].lstrip('.')}=.o)

# Target
TARGET = kernel_port

# Default target
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
"""
    
    def _generate_meson_template(self, backend_config: Dict[str, Any]) -> str:
        """Generate Meson template"""
        return f"""# meson.build - Generated by K_And_S Build System Integrator
project('kernel_porting', 'cpp', default_options: ['cpp_std=c++17'])

# Backend configuration
backend = get_option('backend')

# Source files
sources = files('src\main.cpp')

# Executable
executable('kernel_port', sources, install: true)
"""
    
    def _generate_bazel_template(self, backend_config: Dict[str, Any]) -> str:
        """Generate Bazel template"""
        return f"""# BUILD - Generated by K_And_S Build System Integrator
cc_binary(
    name = "kernel_port",
    srcs = ["src\main.cpp"],
    copts = ["-O3", "-std=c++17"],
    deps = [],
)
"""
    
    def _generate_ninja_template(self, backend_config: Dict[str, Any]) -> str:
        """Generate Ninja template"""
        return f"""# build.ninja - Generated by K_And_S Build System Integrator
cc = {backend_config.get('compiler', 'gcc')}
cflags = -O3 -std=c++17

rule compile
  command = $cc $cflags -c $in -o $out

rule link
  command = $cc $cflags $in -o $out

build src\main.o: compile src\main.cpp
build kernel_port: link src\main.o
"""
    
    def _generate_scons_template(self, backend_config: Dict[str, Any]) -> str:
        """Generate SCons template"""
        return f"""# SConstruct - Generated by K_And_S Build System Integrator
env = Environment(CXX='{backend_config.get('compiler', 'g++')}', CXXFLAGS='-O3 -std=c++17')
env.Program('kernel_port', ['src\main.cpp'])
"""

# ── Tool Registration ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    
    async def test_integrator():
        integrator = BuildSystemIntegrator()
        
        config = {
            "backend": "hip",
            "compiler": "hipcc",
            "extensions": [".hip", ".cu"]
        }
        
        result = await integrator.integrate_build_system(
            "C:\Users\rr\K_And_S",
            "cmake",
            config
        )
        print("Build Integration Result:", result)
    
    asyncio.run(test_integrator())
    print("Build System Integrator initialized successfully!")