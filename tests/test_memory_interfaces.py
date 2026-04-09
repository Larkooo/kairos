from __future__ import annotations

import unittest

from kairos.consolidator import HeuristicConsolidator
from kairos.ranker import HeuristicMemoryRanker
from kairos.store import InMemoryMemoryStore, MemoryQuery
from kairos.types import (
    AgentAction,
    ConsolidatorInput,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    Observation,
    RankerInput,
    TaskState,
)


class KairosInterfacesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.task_state = TaskState(
            task_id="task-auth-expiry",
            episode_id="episode-001",
            user_query="Fix the failing auth expiry test",
            current_goal="Investigate token expiry handling in auth tests",
            subgoals=["inspect auth fixtures", "check token TTL"],
            constraints=["do not break refresh flow"],
            open_loops=["why does the test fail after one hour?"],
            active_files=["tests/test_auth.py", "auth/session.py"],
            active_entities=["token", "expiry", "refresh"],
            current_timestamp=7_200.0,
        )
        self.observation = Observation(
            timestamp=7_200.0,
            user_message="The login test fails after one hour.",
            tool_name="pytest",
            tool_output="AssertionError: token expired",
        )

    def test_in_memory_store_filters_records(self) -> None:
        store = InMemoryMemoryStore()
        first = MemoryRecord(
            memory_id="m1",
            episode_id="episode-001",
            task_id="task-auth-expiry",
            type=MemoryType.FACT,
            status=MemoryStatus.ACTIVE,
            content="Access tokens expire after 3600 seconds.",
            created_at=1_000.0,
            last_accessed_at=1_000.0,
            last_updated_at=1_000.0,
            importance=0.9,
            tags=["auth", "token"],
        )
        second = MemoryRecord(
            memory_id="m2",
            episode_id="episode-002",
            task_id="task-ui",
            type=MemoryType.RESULT,
            status=MemoryStatus.RESOLVED,
            content="Fixed the button color bug.",
            created_at=2_000.0,
            last_accessed_at=2_000.0,
            last_updated_at=2_000.0,
            importance=0.2,
            tags=["ui"],
        )
        store.write([first, second])

        records = store.query(MemoryQuery(task_id="task-auth-expiry"))
        self.assertEqual(["m1"], [record.memory_id for record in records])

    def test_ranker_prefers_task_relevant_memory(self) -> None:
        ranker = HeuristicMemoryRanker()
        candidates = [
            MemoryRecord(
                memory_id="m1",
                episode_id="episode-001",
                task_id="task-auth-expiry",
                type=MemoryType.FACT,
                status=MemoryStatus.ACTIVE,
                content="Access tokens expire after 3600 seconds.",
                created_at=3_600.0,
                last_accessed_at=3_600.0,
                last_updated_at=3_600.0,
                importance=0.9,
                tags=["token", "expiry", "auth"],
            ),
            MemoryRecord(
                memory_id="m2",
                episode_id="episode-003",
                task_id="task-ui",
                type=MemoryType.RESULT,
                status=MemoryStatus.RESOLVED,
                content="The dashboard card color was adjusted last week.",
                created_at=3_000.0,
                last_accessed_at=3_000.0,
                last_updated_at=3_000.0,
                importance=0.1,
                tags=["ui"],
            ),
        ]
        out = ranker.score(
            RankerInput(
                task_state=self.task_state,
                observation=self.observation,
                candidates=candidates,
                top_k=1,
            )
        )
        self.assertEqual(["m1"], out.selected_ids)
        self.assertGreater(out.scored[0].total, out.scored[1].total)

    def test_consolidator_emits_useful_records(self) -> None:
        consolidator = HeuristicConsolidator()
        out = consolidator.consolidate(
            ConsolidatorInput(
                task_state=self.task_state,
                observations=[self.observation],
                actions=[
                    AgentAction(kind="plan_update", text="Inspect auth/session.py expiry math"),
                    AgentAction(kind="tool_call", tool_name="pytest", tool_args={"target": "tests/test_auth.py"}),
                ],
            )
        )
        types = {record.type for record in out.writes}
        self.assertIn(MemoryType.GOAL, types)
        self.assertIn(MemoryType.CONSTRAINT, types)
        self.assertIn(MemoryType.OPEN_LOOP, types)
        self.assertIn(MemoryType.OBSERVATION, types)
        self.assertIn(MemoryType.RESULT, types)
        self.assertIn(MemoryType.PLAN_STEP, types)
        self.assertIn(MemoryType.DECISION, types)
        self.assertEqual([], out.updates)
        self.assertEqual([], out.resolves)


if __name__ == "__main__":
    unittest.main()
