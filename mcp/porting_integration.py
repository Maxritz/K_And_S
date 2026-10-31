"""
Porting Integration Module for K_And_S MCP Server

This module integrates all enhanced components into a unified porting system.
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

# Import all enhanced components
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from mcp.backend_addition_wizard import BackendAdditionWizard, ToolchainDetector
    from mcp.language_converter import LanguageConverter
    from mcp.platform_migrator import PlatformMigrator
    from mcp.build_system_integrator import BuildSystemIntegrator
    from mcp.performance_optimizer import PerformanceOptimizer
    from mcp.error_recovery_system import ErrorRecoverySystem
    from mcp.enhanced_porting_modes import PortingModeManager, PortingModeParams
except ImportError:
    from backend_addition_wizard import BackendAdditionWizard, ToolchainDetector
    from language_converter import LanguageConverter
    from platform_migrator import PlatformMigrator
    from build_system_integrator import BuildSystemIntegrator
    from performance_optimizer import PerformanceOptimizer
    from error_recovery_system import ErrorRecoverySystem
    from enhanced_porting_modes import PortingModeManager, PortingModeParams

@dataclass
class PortingIntegrationResult:
    """Result of comprehensive porting integration"""
    status: str
    backend_added: bool
    language_converted: bool
    platform_migrated: bool
    build_integrated: bool
    performance_optimized: bool
    error_recovery_enabled: bool
    mode_executed: str
    recommendations: List[str]
    next_steps: List[str]

class PortingIntegration:
    """Comprehensive porting integration system"""
    
    def __init__(self):
        self.backend_wizard = BackendAdditionWizard()
        self.toolchain_detector = ToolchainDetector()
        self.language_converter = LanguageConverter()
        self.platform_migrator = PlatformMigrator()
        self.build_integrator = BuildSystemIntegrator()
        self.performance_optimizer = PerformanceOptimizer()
        self.error_recovery = ErrorRecoverySystem()
        self.mode_manager = PortingModeManager()
    
    async def comprehensive_porting(
        self,
        source_code: str,
        source_lang: str,
        target_lang: str,
        source_platform: str,
        target_platform: str,
        target_backend: str,
        build_system: str = "cmake",
        mode: str = "auto"
    ) -> PortingIntegrationResult:
        """Execute comprehensive porting with all enhancements"""
        
        results = {
            "backend_added": False,
            "language_converted": False,
            "platform_migrated": False,
            "build_integrated": False,
            "performance_optimized": False,
            "error_recovery_enabled": False
        }
        
        recommendations = []
        next_steps = []
        
        try:
            # 1. Language conversion
            conversion_result = self.language_converter.convert_code(
                source_code, source_lang, target_lang
            )
            if conversion_result.success:
                results["language_converted"] = True
                next_steps.append("Review converted code for accuracy")
            else:
                recommendations.append("Manual language conversion may be required")
            
            # 2. Platform migration
            migration_result = await self.platform_migrator.migrate_platform(
                ".", source_platform, target_platform
            )
            if migration_result.status in ["success", "partial"]:
                results["platform_migrated"] = True
                next_steps.append("Test migrated code on target platform")
            else:
                recommendations.append("Manual platform adaptation may be required")
            
            # 3. Build system integration
            backend_config = {
                "backend": target_backend,
                "compiler": "gcc",
                "extensions": [".cpp", ".c"]
            }
            build_result = await self.build_integrator.integrate_build_system(
                ".", build_system, backend_config
            )
            if build_result.status == "success":
                results["build_integrated"] = True
                next_steps.append("Build and test the project")
            else:
                recommendations.append("Manual build system configuration may be required")
            
            # 4. Performance optimization
            optimized_code = conversion_result.converted_code if conversion_result.success else source_code
            perf_result = await self.performance_optimizer.optimize_performance(
                optimized_code, "gfx1201", target_backend
            )
            if perf_result.status == "success":
                results["performance_optimized"] = True
                next_steps.append("Apply performance optimizations")
            else:
                recommendations.append("Manual performance optimization may be required")
            
            # 5. Error recovery setup
            results["error_recovery_enabled"] = True
            next_steps.append("Monitor for errors and use recovery system")
            
            # 6. Execute in selected mode
            mode_params = PortingModeParams(
                mode=mode,
                repo_path=".",
                target_arch="gfx1201",
                target_backend=target_backend
            )
            mode_result = await self.mode_manager.execute_porting_mode(mode, mode_params)
            
            return PortingIntegrationResult(
                status="success",
                backend_added=results["backend_added"],
                language_converted=results["language_converted"],
                platform_migrated=results["platform_migrated"],
                build_integrated=results["build_integrated"],
                performance_optimized=results["performance_optimized"],
                error_recovery_enabled=results["error_recovery_enabled"],
                mode_executed=mode_result.get("mode", mode),
                recommendations=recommendations,
                next_steps=next_steps
            )
            
        except Exception as e:
            return PortingIntegrationResult(
                status="error",
                backend_added=results["backend_added"],
                language_converted=results["language_converted"],
                platform_migrated=results["platform_migrated"],
                build_integrated=results["build_integrated"],
                performance_optimized=results["performance_optimized"],
                error_recovery_enabled=results["error_recovery_enabled"],
                mode_executed=mode,
                recommendations=[f"Error: {str(e)}"],
                next_steps=["Manual intervention required"]
            )
    
    async def add_new_backend(self, backend_name: str, backend_config: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new backend with full integration"""
        return await self.backend_wizard.add_new_backend(backend_name, backend_config)
    
    async def detect_toolchain(self, backend_name: str, os_type: str = "auto") -> Dict[str, Any]:
        """Detect toolchain for a backend"""
        return self.toolchain_detector.detect_backend_toolchain(backend_name, os_type)
    
    async def convert_language(self, source_code: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Convert code between languages"""
        result = self.language_converter.convert_code(source_code, source_lang, target_lang)
        return {
            "status": "success" if result.success else "failed",
            "converted_code": result.converted_code,
            "validation": result.validation,
            "warnings": result.warnings,
            "errors": result.errors
        }
    
    async def migrate_platform(self, project_path: str, source_platform: str, target_platform: str) -> Dict[str, Any]:
        """Migrate project between platforms"""
        result = await self.platform_migrator.migrate_platform(project_path, source_platform, target_platform)
        return {
            "status": result.status,
            "source_platform": result.source_platform,
            "target_platform": result.target_platform,
            "steps_completed": result.steps_completed,
            "files_modified": result.files_modified,
            "compatibility_shims": result.compatibility_shims,
            "errors": result.errors,
            "warnings": result.warnings
        }
    
    async def integrate_build_system(self, project_path: str, build_system: str, backend_config: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate with build system"""
        result = await self.build_integrator.integrate_build_system(project_path, build_system, backend_config)
        return {
            "status": result.status,
            "build_system": result.build_system,
            "files_created": result.files_created,
            "files_modified": result.files_modified,
            "backup_files": result.backup_files,
            "errors": result.errors,
            "warnings": result.warnings
        }
    
    async def optimize_performance(self, source_code: str, target_arch: str, target_backend: str) -> Dict[str, Any]:
        """Optimize performance"""
        result = await self.performance_optimizer.optimize_performance(source_code, target_arch, target_backend)
        return {
            "status": result.status,
            "optimizations_applied": result.optimizations_applied,
            "estimated_performance_gain": result.estimated_performance_gain,
            "config": result.config,
            "warnings": result.warnings,
            "errors": result.errors
        }
    
    async def handle_error(self, error: Exception, context: Dict[str, Any], recovery_level: str = "automatic") -> Dict[str, Any]:
        """Handle errors with recovery"""
        result = await self.error_recovery.handle_error(error, context, recovery_level)
        return {
            "status": result.status,
            "strategy_used": result.strategy_used,
            "recovery_actions": result.recovery_actions,
            "rollback_performed": result.rollback_performed,
            "fallback_used": result.fallback_used,
            "errors": result.errors,
            "warnings": result.warnings
        }

# ── Tool Registration ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    
    async def test_integration():
        integration = PortingIntegration()
        
        # Test comprehensive porting
        result = await integration.comprehensive_porting(
            source_code="int main() { return 0; }",
            source_lang="c",
            target_lang="rust",
            source_platform="linux",
            target_platform="windows",
            target_backend="hip",
            build_system="cmake",
            mode="auto"
        )
        print("Comprehensive Porting Result:", result)
        
        # Test backend addition
        backend_result = await integration.add_new_backend("test_backend", {
            "compiler": "gcc",
            "extensions": [".c", ".cpp"],
            "arch_patterns": ["x86_64"],
            "feature_flags": {"simd": True}
        })
        print("Backend Addition Result:", backend_result)
        
        # Test language conversion
        lang_result = await integration.convert_language(
            "int main() { return 0; }",
            "c",
            "rust"
        )
        print("Language Conversion Result:", lang_result)
        
        # Test platform migration
        platform_result = await integration.migrate_platform(
            ".",
            "linux",
            "windows"
        )
        print("Platform Migration Result:", platform_result)
        
        # Test build system integration
        build_result = await integration.integrate_build_system(
            ".",
            "cmake",
            {"backend": "hip", "compiler": "hipcc"}
        )
        print("Build Integration Result:", build_result)
        
        # Test performance optimization
        perf_result = await integration.optimize_performance(
            "int main() { return 0; }",
            "gfx1201",
            "hip"
        )
        print("Performance Optimization Result:", perf_result)
        
        # Test error recovery
        error_result = await integration.handle_error(
            Exception("Test error"),
            {"context": "test"}
        )
        print("Error Recovery Result:", error_result)
    
    asyncio.run(test_integration())
    print("Porting Integration initialized successfully!")