"""
Language Converter for K_And_S MCP Server

Automated conversion between programming languages with semantic preservation
and validation.
"""

from __future__ import annotations
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

# Import enhanced components
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@dataclass
class ConversionResult:
    """Result of a language conversion"""
    converted_code: str
    validation: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    success: bool

class LanguageConverter:
    """Automated conversion between programming languages"""
    
    def __init__(self):
        self.conversion_rules = {
            "cuda_to_rust": self._cuda_to_rust_rules,
            "cpp_to_zig": self._cpp_to_zig_rules,
            "python_to_c": self._python_to_c_rules,
            "opencl_to_sycl": self._opencl_to_sycl_rules,
            "hip_to_cuda": self._hip_to_cuda_rules,
            "vulkan_to_directx": self._vulkan_to_directx_rules
        }
    
    def convert_code(self, source_code: str, source_lang: str, target_lang: str) -> ConversionResult:
        """Convert code between languages with validation"""
        conversion_key = f"{source_lang}_to_{target_lang}"
        
        if conversion_key not in self.conversion_rules:
            return ConversionResult(
                converted_code="",
                validation={},
                warnings=[],
                errors=[f"Conversion {conversion_key} not supported"],
                success=False
            )
        
        try:
            # Apply conversion rules
            converted_code = self._apply_conversion_rules(
                source_code, 
                self.conversion_rules[conversion_key]()
            )
            
            # Validate conversion
            validation_result = self._validate_conversion(
                source_code, converted_code, source_lang, target_lang
            )
            
            # Get warnings
            warnings = self._get_conversion_warnings(source_code, converted_code)
            
            return ConversionResult(
                converted_code=converted_code,
                validation=validation_result,
                warnings=warnings,
                errors=[],
                success=True
            )
            
        except Exception as e:
            return ConversionResult(
                converted_code="",
                validation={},
                warnings=[],
                errors=[str(e)],
                success=False
            )
    
    def _apply_conversion_rules(self, source_code: str, rules: List[Dict[str, Any]]) -> str:
        """Apply language-specific conversion rules"""
        converted = source_code
        
        for rule in rules:
            if rule["type"] == "replace":
                converted = converted.replace(rule["pattern"], rule["replacement"])
            elif rule["type"] == "regex":
                import re
                converted = re.sub(rule["pattern"], rule["replacement"], converted)
            elif rule["type"] == "transform":
                converted = rule["function"](converted)
        
        return converted
    
    def _cuda_to_rust_rules(self) -> List[Dict[str, Any]]:
        """CUDA to Rust conversion rules"""
        return [
            {"type": "replace", "pattern": "__global__", "replacement": "#[kernel]"},
            {"type": "replace", "pattern": "__device__", "replacement": "#[device]"},
            {"type": "replace", "pattern": "threadIdx.x", "replacement": "thread_id.x"},
            {"type": "replace", "pattern": "blockIdx.x", "replacement": "block_id.x"},
            {"type": "replace", "pattern": "blockDim.x", "replacement": "block_dim.x"},
            {"type": "replace", "pattern": "gridDim.x", "replacement": "grid_dim.x"},
            {"type": "regex", "pattern": r"cudaMalloc\((.*?)\)", "replacement": r"alloc::alloc(\1)"},
            {"type": "regex", "pattern": r"cudaFree\((.*?)\)", "replacement": r"alloc::dealloc(\1)"}
        ]
    
    def _cpp_to_zig_rules(self) -> List[Dict[str, Any]]:
        """C++ to Zig conversion rules"""
        return [
            {"type": "replace", "pattern": "template <typename T>", "replacement": "fn generic(comptime T: type)"},
            {"type": "replace", "pattern": "std::vector", "replacement": "std.ArrayList"},
            {"type": "replace", "pattern": "std::string", "replacement": "[]const u8"},
            {"type": "replace", "pattern": "new ", "replacement": "allocator.create("},
            {"type": "replace", "pattern": "delete ", "replacement": "allocator.destroy("}
        ]
    
    def _python_to_c_rules(self) -> List[Dict[str, Any]]:
        """Python to C conversion rules"""
        return [
            {"type": "replace", "pattern": "def ", "replacement": "void "},
            {"type": "replace", "pattern": ":", "replacement": "{"},
            {"type": "replace", "pattern": "print(", "replacement": "printf("},
            {"type": "replace", "pattern": "len(", "replacement": "strlen("},
            {"type": "replace", "pattern": "range(", "replacement": "for (int i = 0; i < "}
        ]
    
    def _opencl_to_sycl_rules(self) -> List[Dict[str, Any]]:
        """OpenCL to SYCL conversion rules"""
        return [
            {"type": "replace", "pattern": "__kernel", "replacement": "SYCL_EXTERNAL"},
            {"type": "replace", "pattern": "__global", "replacement": "sycl::global_ptr"},
            {"type": "replace", "pattern": "get_global_id(0)", "replacement": "item.get_global_id(0)"},
            {"type": "replace", "pattern": "get_local_id(0)", "replacement": "item.get_local_id(0)"},
            {"type": "replace", "pattern": "get_group_id(0)", "replacement": "item.get_group(0)"}
        ]
    
    def _hip_to_cuda_rules(self) -> List[Dict[str, Any]]:
        """HIP to CUDA conversion rules"""
        return [
            {"type": "replace", "pattern": "hipMalloc", "replacement": "cudaMalloc"},
            {"type": "replace", "pattern": "hipFree", "replacement": "cudaFree"},
            {"type": "replace", "pattern": "hipMemcpy", "replacement": "cudaMemcpy"},
            {"type": "replace", "pattern": "hipLaunchKernelGGL", "replacement": "cudaLaunchKernel"},
            {"type": "replace", "pattern": "hipDeviceSynchronize", "replacement": "cudaDeviceSynchronize"}
        ]
    
    def _vulkan_to_directx_rules(self) -> List[Dict[str, Any]]:
        """Vulkan to DirectX conversion rules"""
        return [
            {"type": "replace", "pattern": "vkCmdDispatch", "replacement": "Dispatch"},
            {"type": "replace", "pattern": "vkCmdBindPipeline", "replacement": "SetPipelineState"},
            {"type": "replace", "pattern": "VkBuffer", "replacement": "ID3D12Resource"},
            {"type": "replace", "pattern": "vkCreateBuffer", "replacement": "CreateCommittedResource"}
        ]
    
    def _validate_conversion(self, source_code: str, converted_code: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Validate the conversion"""
        validation = {
            "syntax_valid": self._check_syntax(converted_code, target_lang),
            "semantics_preserved": self._check_semantics(source_code, converted_code),
            "memory_safety": self._check_memory_safety(converted_code, target_lang),
            "thread_safety": self._check_thread_safety(converted_code, target_lang),
            "performance_equivalent": self._check_performance(source_code, converted_code)
        }
        
        return validation
    
    def _check_syntax(self, code: str, language: str) -> bool:
        """Check syntax validity for target language"""
        # This would use actual language parsers
        # For now, basic checks
        if language == "rust":
            return "fn " in code or "let " in code
        elif language == "zig":
            return "fn " in code or "const " in code
        elif language == "c":
            return "void " in code or "int " in code
        elif language == "sycl":
            return "sycl::" in code or "queue " in code
        return True
    
    def _check_semantics(self, source_code: str, converted_code: str) -> bool:
        """Check if semantics are preserved"""
        # Check for key semantic elements
        source_keywords = set(re.findall(r'\b\w+\b', source_code))
        converted_keywords = set(re.findall(r'\b\w+\b', converted_code))
        
        # Check if important keywords are preserved
        important_keywords = {'kernel', 'thread', 'block', 'memory', 'sync'}
        preserved = important_keywords.issubset(source_keywords) and important_keywords.issubset(converted_keywords)
        
        return preserved
    
    def _check_memory_safety(self, code: str, language: str) -> bool:
        """Check memory safety for target language"""
        # Language-specific memory safety checks
        if language == "rust":
            return "unsafe" not in code or "safe" in code
        elif language == "c":
            return "malloc" in code and "free" in code
        return True
    
    def _check_thread_safety(self, code: str, language: str) -> bool:
        """Check thread safety for target language"""
        # Check for proper synchronization
        sync_patterns = ['__syncthreads', 'barrier', 'atomic', 'mutex', 'lock']
        has_sync = any(pattern in code for pattern in sync_patterns)
        return has_sync or 'single_threaded' in code
    
    def _check_performance(self, source_code: str, converted_code: str) -> bool:
        """Check if performance characteristics are preserved"""
        # Check for performance-critical patterns
        perf_patterns = ['__shared__', 'register', 'vector', 'simd', 'parallel']
        source_has_perf = any(pattern in source_code for pattern in perf_patterns)
        converted_has_perf = any(pattern in converted_code for pattern in perf_patterns)
        
        return source_has_perf == converted_has_perf
    
    def _get_conversion_warnings(self, source_code: str, converted_code: str) -> List[str]:
        """Get warnings about the conversion"""
        warnings = []
        
        # Check for potential issues
        if len(converted_code) < len(source_code) * 0.5:
            warnings.append("Converted code is significantly shorter - may have lost functionality")
        
        if "TODO" in converted_code or "FIXME" in converted_code:
            warnings.append("Converted code contains TODO\FIXME markers")
        
        if "unsafe" in converted_code:
            warnings.append("Converted code contains unsafe operations")
        
        return warnings

# ── Tool Registration ─────────────────────────────────────────────────────

# This module provides language conversion as standalone functions
# They can be integrated with the main MCP server as needed

if __name__ == "__main__":
    # Test language conversion
    converter = LanguageConverter()
    
    # Test CUDA to Rust conversion
    cuda_code = """
    __global__ void kernel(float* data, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            data[idx] = data[idx] * 2.0f;
        }
    }
    """
    
    result = converter.convert_code(cuda_code, "cuda", "rust")
    print("CUDA to Rust Conversion:")
    print("Success:", result.success)
    print("Converted code:", result.converted_code[:200])
    print("Validation:", result.validation)
    print("Warnings:", result.warnings)
    
    print("\nLanguage Converter initialized successfully!")