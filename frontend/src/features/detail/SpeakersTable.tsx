// One row per speaker, remote ones included (UI doc, "Speakers and positions table"). It is also the plan's text
// equivalent: every speaker's position, distance and facing as text.

import { SortableTable, type Column } from "../../components/SortableTable";
import { Bar } from "../../components/ui";
import type { MeetingSpec } from "../../lib/api";
import { DASH, metres, pct, signedDeg } from "../../lib/format";
import type { PersonRow } from "../../lib/people";

export function SpeakersTable({ m, rows, selected, onSelect }: { m: MeetingSpec; rows: PersonRow[]; selected: string; onSelect: (id: string) => void }) {
  const max = Math.max(...rows.map((r) => r.person.turn_share), 0.01);
  const columns: Column<PersonRow>[] = [
    {
      key: "speaker",
      header: "Speaker",
      sort: (r) => r.id,
      cell: (r) => (
        <button
          type="button"
          aria-pressed={r.id === selected}
          aria-label={`Select ${r.id}, ${r.person.persona.role}`}
          onClick={() => onSelect(r.id)}
          className="bg-transparent border-0 p-0 num text-focus cursor-pointer hover:underline"
        >
          {r.id}
        </button>
      ),
    },
    { key: "position", header: "Position", sort: (r) => r.position, cell: (r) => r.position },
    { key: "persona", header: "Persona", sort: (r) => r.person.persona.role, cell: (r) => `${r.person.persona.role}${r.person.is_chair ? " · chair" : ""}` },
    { key: "l1", header: "L1 · code-switching", sort: (r) => r.person.persona.l1, cell: (r) => `${r.person.persona.l1} · ${r.person.persona.code_switching}` },
    {
      key: "share",
      header: "Turn share",
      sort: (r) => r.person.turn_share,
      cell: (r) => (
        <span className="flex items-center gap-2 justify-end">
          <Bar value={r.person.turn_share} max={max} className="w-24" />
          <span className="num w-12 text-right">{pct(r.person.turn_share)}</span>
        </span>
      ),
      align: "right",
    },
    { key: "dist", header: "Mic distance", sort: (r) => r.micDistance ?? 99, cell: (r) => (r.micDistance === null ? "speakerphone" : metres(r.micDistance)), align: "right" },
    { key: "facing", header: "Facing", sort: (r) => r.facing ?? 999, cell: (r) => (r.person.position_type === "remote" ? DASH : signedDeg(r.facing)), align: "right" },
    { key: "nsc", header: "NSC reference", sort: (r) => r.nsc, cell: (r) => <span className="num">{`P3 · spk ${r.nsc}${r.swapped ? " *" : ""}`}</span> },
  ];
  const n = (t: string) => m.people.filter((p) => p.position_type === t).length;
  const swapped = rows.some((r) => r.swapped);
  return (
    <SortableTable
      rows={rows}
      columns={columns}
      caption={`Speakers and positions in ${m.meeting_id}`}
      rowKey={(r) => r.id}
      rowClass={(r) => r.id === selected && "bg-active/60"}
      footer={`${rows.length} of ${rows.length} shown · ${n("table")} at table · ${n("wall")} at wall · ${n("presenter")} presenter · ${n("remote")} remote${swapped ? " · * voice swapped in stage 4" : ""}`}
    />
  );
}
