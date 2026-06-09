"""Tests for the tamper-evident audit log."""

from govgridmarl.governance.audit_log import AuditLog


def test_audit_chain_verifies():
    log = AuditLog()
    log.append("household_0", decision=0.2, constraint_checks={"voltage": "ok"})
    log.append("household_1", decision=0.0, constraint_checks={"voltage": "ok"})
    assert len(log) == 2
    assert log.verify() is True


def test_tampering_breaks_chain():
    log = AuditLog()
    log.append("household_0", decision=0.2, constraint_checks={"voltage": "ok"})
    # mutate a record in place -> digest no longer matches stored hash
    log._records[0][0].decision = 0.9
    assert log.verify() is False
