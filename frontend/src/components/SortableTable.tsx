// A small sortable table in the AeroAudio style (DESIGN_SYSTEM §7 SortableTable): sticky header, sort buttons with
// aria-sort, right-aligned numeric columns, "N of M shown" under the frame. Tables here are small (one row per
// speaker or per histogram bin), so there is no virtualisation.

import { useState, type ReactNode } from "react";
import { cx } from "./ui";

export interface Column<T> {
  key: string;
  header: string;
  cell: (row: T) => ReactNode;
  sort?: (row: T) => number | string;
  align?: "right";
}

export function SortableTable<T>({ rows, columns, caption, rowKey, rowClass, initialSort, footer }: {
  rows: T[];
  columns: Column<T>[];
  caption: string;
  rowKey: (row: T) => string;
  rowClass?: (row: T) => string | false | undefined;
  initialSort?: { key: string; desc: boolean };
  footer?: ReactNode;
}) {
  const [sort, setSort] = useState(initialSort ?? null);
  const col = columns.find((c) => c.key === sort?.key);
  const sorted = col?.sort
    ? [...rows].sort((a, b) => {
        const x = col.sort!(a);
        const y = col.sort!(b);
        const d = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y));
        return sort!.desc ? -d : d;
      })
    : rows;
  return (
    <div className="flex flex-col gap-2 min-w-0">
      <div className="border border-track rounded-md overflow-auto scroll-thin max-h-[70vh]">
        <table className="w-full min-w-[760px] text-[13px] border-collapse">
          <caption className="sr-only">{caption}</caption>
          <thead className="sticky top-0 bg-panel">
            <tr className="border-b border-line">
              {columns.map((c) => {
                const active = sort?.key === c.key;
                return (
                  <th
                    key={c.key}
                    scope="col"
                    aria-sort={active ? (sort!.desc ? "descending" : "ascending") : undefined}
                    className={cx("px-3 py-2 text-[11px] font-medium text-muted uppercase tracking-wider whitespace-nowrap", c.align === "right" ? "text-right" : "text-left")}
                  >
                    {c.sort ? (
                      <button
                        type="button"
                        onClick={() => setSort({ key: c.key, desc: active ? !sort!.desc : false })}
                        className="bg-transparent border-0 p-0 text-inherit uppercase tracking-wider cursor-pointer hover:text-fg"
                      >
                        {c.header} <span aria-hidden>{active ? (sort!.desc ? "▼" : "▲") : "↕"}</span>
                      </button>
                    ) : (
                      c.header
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => (
              <tr key={rowKey(r)} className={cx("border-b border-track last:border-0 hover:bg-track/60", rowClass?.(r))}>
                {columns.map((c) => (
                  <td key={c.key} className={cx("px-3 py-2 whitespace-nowrap", c.align === "right" && "text-right num")}>
                    {c.cell(r)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {footer && <span className="text-xs text-muted num">{footer}</span>}
    </div>
  );
}
