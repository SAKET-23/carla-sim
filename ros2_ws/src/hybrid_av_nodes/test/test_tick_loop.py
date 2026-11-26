import threading
import time

from hybrid_av_nodes.tick_loop import CommandMailbox, TickLoop

from hybrid_av.types import ActuatorCommand


def make(n_steps, responder=None, timeout=0.2, warmup=0):
    applied, mb, t = [], CommandMailbox(), [0.0]

    def step(cmd):
        applied.append(cmd)
        t[0] += 0.05
        return t[0], len(applied) >= n_steps

    def publish(stamp):
        if responder:
            threading.Thread(target=responder, args=(mb, stamp)).start()

    return TickLoop(step, publish, mb, command_timeout=timeout, warmup_ticks=warmup), applied


def test_tick_loop_waits_for_matching_stamp():
    def responder(mb, stamp):
        time.sleep(0.02)
        mb.put(stamp - 0.05, ActuatorCommand(0, 0, 0))  # stale stamp: ignored
        mb.put(stamp, ActuatorCommand(0.1, 0.5, 0.0))

    loop, applied = make(3, responder)
    stats = loop.run()
    assert stats.ticks == 3 and stats.late_commands == 0
    assert applied[0] == ActuatorCommand(0.0, 0.0, 1.0) and applied[1] == ActuatorCommand(0.1, 0.5, 0.0)


def test_late_command_counted_after_warmup():
    loop, applied = make(4, responder=None, timeout=0.01, warmup=2)
    stats = loop.run()
    assert stats.late_commands == 2 and stats.warmup_ticks == 2
    assert all(c == ActuatorCommand(0.0, 0.0, 1.0) for c in applied)  # last command reused


def test_max_ticks():
    loop, _ = make(100, responder=lambda mb, s: mb.put(s, ActuatorCommand(0, 0, 0)))
    assert loop.run(max_ticks=5).ticks == 5
