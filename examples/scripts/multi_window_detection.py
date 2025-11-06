"""Example: Multi-window STA/LTA detection with filtering."""

from seiscae import EventDetector, SpectrogramGenerator, EventExtractor, EventCatalog

# Example 1: Multi-window with dead time and minimum duration
print("Example 1: Multi-window STA/LTA with event filtering")
detector = EventDetector.create_multi_window(
    sta_seconds=0.1,
    lta_seconds=30.0,
    delta_sta=20.0,
    delta_lta=20.0,
    epsilon=1.8,
    threshold_on=1.1,
    threshold_off=1.008,
    min_event_duration=0.001,  # Filter events shorter than 1ms
    dead_time=1.2               # 1.2 seconds between events
)

# Detect events
st, cft, triggers = detector.detect_file("path/to/your/data.mseed")
print(f"Detected {len(triggers)} events using multi-window STA/LTA")
print(f"  - Min event duration: {detector.min_event_duration}s")
print(f"  - Dead time: {detector.dead_time}s")

# Example 2: Classic with filtering
print("\nExample 2: Classic STA/LTA with filtering")
classic_detector = EventDetector.create_classic(
    min_event_duration=1.0,
    dead_time=5.0
)

st_c, cft_c, triggers_c = classic_detector.detect_file("path/to/your/data.mseed")
print(f"Detected {len(triggers_c)} events using classic STA/LTA")

# Example 3: Full pipeline with catalog
print("\nExample 3: Full pipeline with catalog")
spec_gen = SpectrogramGenerator(freq_min=1.0, freq_max=50.0)
extractor = EventExtractor(spec_gen)
events = extractor.extract_events(st[0], triggers)

# Add detection metadata to events
for event in events:
    event['detection_method'] = detector.method
    event['min_event_duration'] = detector.min_event_duration
    event['dead_time'] = detector.dead_time

catalog = EventCatalog(events)
catalog.save("filtered_catalog.pkl")
print(f"Saved {len(catalog)} events to catalog")

# Example 4: Comparison with and without filtering
print("\nExample 4: Impact of filtering")

# Without filtering
detector_no_filter = EventDetector.create_multi_window(
    threshold_on=1.1,
    threshold_off=1.008,
    min_event_duration=0.0,
    dead_time=0.0
)
st1, cft1, triggers1 = detector_no_filter.detect_file("path/to/your/data.mseed")

# With filtering
detector_with_filter = EventDetector.create_multi_window(
    threshold_on=1.1,
    threshold_off=1.008,
    min_event_duration=0.5,
    dead_time=1.2
)
st2, cft2, triggers2 = detector_with_filter.detect_file("path/to/your/data.mseed")

print(f"Without filtering: {len(triggers1)} events")
print(f"With filtering: {len(triggers2)} events")
print(f"Filtered out: {len(triggers1) - len(triggers2)} events")

# Example 5: Using constructor directly
print("\nExample 5: Using constructor with all parameters")
detector_custom = EventDetector(
    sta_seconds=0.1,
    lta_seconds=30.0,
    threshold_on=1.1,
    threshold_off=1.008,
    method="multi_window",
    delta_sta=20.0,
    delta_lta=20.0,
    epsilon=1.8,
    min_event_duration=0.001,
    dead_time=1.2,
    highpass_freq=1.0
)

st_custom, cft_custom, triggers_custom = detector_custom.detect_file("path/to/your/data.mseed")
print(f"Custom detector found: {len(triggers_custom)} events")
