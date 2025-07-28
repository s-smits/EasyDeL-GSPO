#!/usr/bin/env python3
"""
Working TPU monitoring using the official LibTPU SDK
"""

from libtpu.sdk import tpumonitoring
import time
import subprocess
import sys

def get_tpu_metrics():
    """Get TPU metrics using LibTPU SDK"""
    print("🔍 TPU Metrics (via LibTPU SDK)")
    print("=" * 50)
    
    try:
        # List all available metrics
        supported_metrics = tpumonitoring.list_supported_metrics()
        print(f"📊 Available metrics: {supported_metrics}")
        
        # Key metrics to monitor
        key_metrics = ['duty_cycle_pct', 'hbm_capacity_usage', 'hbm_capacity_total', 'tensorcore_util']
        
        for metric_name in key_metrics:
            if metric_name in supported_metrics:
                try:
                    metric = tpumonitoring.get_metric(metric_name)
                    print(f"\n📈 {metric_name.upper()}:")
                    print(f"   Description: {metric.description()}")
                    print(f"   Data: {metric.data()}")
                except Exception as e:
                    print(f"   ❌ Error getting {metric_name}: {e}")
            else:
                print(f"   ⚠️  {metric_name} not supported")
                
    except Exception as e:
        print(f"❌ LibTPU monitoring failed: {e}")
        return False
    
    return True

def get_tpu_info_safe():
    """Get tpu-info output safely (ignoring crashes)"""
    print("\n🖥️  TPU-Info Output")
    print("=" * 50)
    
    try:
        # Run tpu-info but capture both stdout and stderr
        result = subprocess.run(
            ['tpu-info'], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.stdout:
            print(result.stdout)
            return True
        elif result.stderr and "TPU" in result.stderr:
            print("Output from stderr:")
            print(result.stderr)
            return True
        else:
            print("❌ No TPU info available")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ tpu-info timed out")
        return False
    except Exception as e:
        print(f"❌ tpu-info failed: {e}")
        return False

def monitor_continuously():
    """Monitor TPUs continuously"""
    print("🔄 Starting continuous TPU monitoring...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            print(f"\n⏰ {time.strftime('%H:%M:%S')} - TPU Status Check")
            print("-" * 60)
            
            # Try LibTPU metrics first
            sdk_success = get_tpu_metrics()
            
            # Then try tpu-info for visual output
            if not sdk_success:
                get_tpu_info_safe()
            
            print("\n" + "="*60)
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")

def main():
    """Main monitoring function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        monitor_continuously()
    else:
        # Single check
        print("🚀 TPU Monitoring (Single Check)")
        print("=" * 60)
        
        # Try both methods
        sdk_success = get_tpu_metrics()
        
        print("\n" + "="*60)
        get_tpu_info_safe()
        
        print("\n💡 For continuous monitoring:")
        print("python tpu_monitor_fixed.py --continuous")

if __name__ == "__main__":
    main()