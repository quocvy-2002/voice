import pytest

from omnivoice.utils.task_control import RunRegistry, TaskCancelled


def test_stop_is_scoped_to_session_and_workflow():
    registry = RunRegistry()
    first = registry.begin("browser-a", "transcribe")
    other_session = registry.begin("browser-b", "transcribe")
    other_workflow = registry.begin("browser-a", "clone")

    assert registry.request_stop("browser-a", "transcribe", first.run_id)
    with pytest.raises(TaskCancelled):
        first.check_cancelled()
    other_session.check_cancelled()
    other_workflow.check_cancelled()
    assert first.snapshot().state == "cancelling"


def test_progress_is_monotonic_and_completion_owns_one_hundred_percent():
    registry = RunRegistry()
    run = registry.begin("browser-a", "clone")
    run.report(0.7, "Đang tạo giọng")
    run.report(0.3, "Đang tạo giọng")
    run.report(1.0, "Đang hoàn tất")

    assert run.snapshot().fraction == 0.99
    assert registry.finish("browser-a", "clone", run.run_id, "completed")
    assert run.snapshot().fraction == 1.0
    assert run.snapshot().state == "completed"


def test_duplicate_start_and_stale_finish_are_ignored():
    registry = RunRegistry()
    first = registry.begin("browser-a", "design")

    assert registry.begin("browser-a", "design") is None
    assert registry.finish("browser-a", "design", first.run_id, "failed")
    second = registry.begin("browser-a", "design")
    assert second.run_id != first.run_id
    assert registry.get("browser-a", "design", first.run_id) is None
    assert not registry.finish("browser-a", "design", first.run_id, "completed")
    assert registry.snapshot("browser-a", "design").run_id == second.run_id


def test_one_run_can_only_be_executed_once():
    run = RunRegistry().begin("browser-a", "clone")

    assert run.claim_execution()
    assert not run.claim_execution()


def test_drop_session_requests_stop_for_all_its_active_runs():
    registry = RunRegistry()
    first = registry.begin("browser-a", "clone")
    second = registry.begin("browser-a", "transcribe")
    untouched = registry.begin("browser-b", "clone")

    registry.drop_session("browser-a")

    with pytest.raises(TaskCancelled):
        first.check_cancelled()
    with pytest.raises(TaskCancelled):
        second.check_cancelled()
    untouched.check_cancelled()
    assert registry.snapshot("browser-a", "clone") is None
