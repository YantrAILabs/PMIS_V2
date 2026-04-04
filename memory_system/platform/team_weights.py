"""
Team Weight Computation — On-demand, no background task.
Called on every /retrieve. Fast for small teams (<10ms for 600 rows).
"""

import json


def recompute_team_weights(conn):
    """Recompute team_weight for all nodes. Bottom-up propagation."""
    team_size = conn.execute(
        "SELECT COUNT(DISTINCT author) FROM nodes WHERE author != ''"
    ).fetchone()[0]
    team_size = max(team_size, 1)

    # Compute anchor weights from pulls + breadth + ratings
    anchors = conn.execute("""
        SELECT id, pull_count, unique_pullers, rating_sum, rating_count
        FROM nodes WHERE type='anchor'
    """).fetchall()

    for a in anchors:
        pull_count = a["pull_count"] or 0
        pull_score = min(pull_count / 20.0, 1.0)

        pullers = []
        try:
            pullers = json.loads(a["unique_pullers"] or "[]")
        except (json.JSONDecodeError, TypeError):
            pass
        breadth_score = len(pullers) / team_size

        rating_count = a["rating_count"] or 0
        rating_sum = a["rating_sum"] or 0.0
        if rating_count > 0:
            rating_score = (rating_sum / rating_count) / 5.0
        else:
            rating_score = 0.5  # neutral until rated

        tw = 0.4 * pull_score + 0.3 * breadth_score + 0.3 * rating_score
        conn.execute("UPDATE nodes SET team_weight=? WHERE id=?",
                     (round(tw, 3), a["id"]))

    # Propagate bottom-up: context = avg(child anchors)
    for ctx in conn.execute("SELECT id FROM nodes WHERE type='context'").fetchall():
        avg_w = conn.execute("""
            SELECT AVG(n.team_weight) FROM nodes n
            JOIN edges e ON e.tgt=n.id
            WHERE e.src=? AND n.type='anchor'
        """, (ctx["id"],)).fetchone()[0]
        conn.execute("UPDATE nodes SET team_weight=? WHERE id=?",
                     (round(avg_w or 0.5, 3), ctx["id"]))

    # Propagate: SC = avg(child contexts)
    for sc in conn.execute("SELECT id FROM nodes WHERE type='super_context'").fetchall():
        avg_w = conn.execute("""
            SELECT AVG(n.team_weight) FROM nodes n
            JOIN edges e ON e.tgt=n.id
            WHERE e.src=? AND n.type='context'
        """, (sc["id"],)).fetchone()[0]
        conn.execute("UPDATE nodes SET team_weight=? WHERE id=?",
                     (round(avg_w or 0.5, 3), sc["id"]))

    conn.commit()
