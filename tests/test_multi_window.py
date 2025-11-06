def test_dead_time_filter():
    """Test dead time filtering."""
    # Create synthetic trace with multiple events
    sampling_rate = 100.0
    duration = 20.0
    npts = int(sampling_rate * duration)
    
    # Create data with 3 events: at 5s, 6s (too close), and 10s
    data = np.random.randn(npts) * 0.1
    
    # Event 1 at 5s
    event1_start = int(5 * sampling_rate)
    data[event1_start:event1_start + 50] += 2.0
    
    # Event 2 at 6s (only 1s after event 1)
    event2_start = int(6 * sampling_rate)
    data[event2_start:event2_start + 50] += 2.0
    
    # Event 3 at 10s (4s after event 1)
    event3_start = int(10 * sampling_rate)
    data[event3_start:event3_start + 50] += 2.0
    
    # Create detector with 2s dead time
    detector = EventDetector.create_multi_window(
        threshold_on=1.5,
        threshold_off=1.0,
        dead_time=2.0
    )
    
    # Manually create triggers
    from obspy import Trace, UTCDateTime
    tr = Trace(data=data)
    tr.stats.sampling_rate = sampling_rate
    tr.stats.starttime = UTCDateTime(0)
    
    # Simulate triggers
    triggers = np.array([
        [event1_start, event1_start + 50],
        [event2_start, event2_start + 50],
        [event3_start, event3_start + 50]
    ])
    
    # Apply filters
    filtered = detector._apply_filters(triggers, sampling_rate)
    
    # Should filter out event 2 (too close to event 1)
    assert len(filtered) == 2, f"Expected 2 events after dead time filter, got {len(filtered)}"


def test_min_event_duration_filter():
    """Test minimum event duration filtering."""
    sampling_rate = 100.0
    
    # Create detector with 0.5s minimum duration
    detector = EventDetector.create_multi_window(
        min_event_duration=0.5
    )
    
    # Create triggers with different durations
    triggers = np.array([
        [100, 120],   # 0.2s (too short)
        [200, 260],   # 0.6s (ok)
        [300, 310],   # 0.1s (too short)
        [400, 500],   # 1.0s (ok)
    ])
    
    # Apply filters
    filtered = detector._apply_filters(triggers, sampling_rate)
    
    # Should keep only the 2 longer events
    assert len(filtered) == 2, f"Expected 2 events after duration filter, got {len(filtered)}"
    assert filtered[0][0] == 200  # First kept event
    assert filtered[1][0] == 400  # Second kept event


def test_combined_filters():
    """Test combining min_event_duration and dead_time filters."""
    sampling_rate = 100.0
    
    detector = EventDetector.create_multi_window(
        min_event_duration=0.3,
        dead_time=1.0
    )
    
    # Create triggers
    triggers = np.array([
        [100, 150],   # 0.5s, ok
        [160, 180],   # 0.2s, too short
        [200, 251],   # 0.51s, ok but too close to first
        [400, 460],   # 0.6s, ok and far enough
    ])
    
    # Apply filters
    filtered = detector._apply_filters(triggers, sampling_rate)
    
    # Should keep events at 100 and 400
    assert len(filtered) == 2
    assert filtered[0][0] == 100
    assert filtered[1][0] == 400


def test_no_filtering():
    """Test that no filtering occurs when parameters are 0."""
    sampling_rate = 100.0
    
    detector = EventDetector.create_multi_window(
        min_event_duration=0.0,
        dead_time=0.0
    )
    
    triggers = np.array([
        [100, 105],
        [106, 110],
        [111, 115],
    ])
    
    filtered = detector._apply_filters(triggers, sampling_rate)
    
    # All triggers should be kept
    assert len(filtered) == len(triggers)
    np.testing.assert_array_equal(filtered, triggers)


def test_parameter_validation_filtering():
    """Test validation of filtering parameters."""
    # Negative min_event_duration
    with pytest.raises(ValueError, match="min_event_duration must be non-negative"):
        EventDetector(min_event_duration=-1.0)
    
    # Negative dead_time
    with pytest.raises(ValueError, match="dead_time must be non-negative"):
        EventDetector(dead_time=-1.0)
