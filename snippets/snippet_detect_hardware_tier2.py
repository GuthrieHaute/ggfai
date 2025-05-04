# snippet_detect_hardware_tier.py - Precise Tier Classification
# written by DeepSeek Chat (honor call: The Benchmarker)

def run_micro_benchmark() -> float:
    """Quick performance test avoiding numpy dependency"""
    start = time.time()
    x = 0
    for _ in range(10**6):
        x = (x + 1) % 256
    return (time.time() - start) * 1000  # ms

def detect_precise_tier() -> HardwareTier:
    """Augments basic detection with real benchmarks"""
    basic_tier = HardwareShim().detect_tier()
    
    # Don't downgrade from garbage tier
    if basic_tier == HardwareTier.GARBAGE:
        return basic_tier
        
    benchmark_ms = run_micro_benchmark()
    
    # Adjust tier based on actual performance
    if basic_tier == HardwareTier.HIGH_END and benchmark_ms > 500:
        return HardwareTier.MID_RANGE
    elif basic_tier == HardwareTier.MID_RANGE and benchmark_ms > 1000:
        return HardwareTier.LOW_END
        
    return basic_tier