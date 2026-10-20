import { createBrowserRouter, Navigate, useParams } from "react-router-dom";
import { Empty, ErrorNote, PageHeader, Spinner } from "../components/ui";
import { MeetingBrowser } from "../features/browser/MeetingBrowser";
import { MeetingDetail } from "../features/detail/MeetingDetail";
import { Distributions } from "../features/distributions/Distributions";
import { useRuns } from "../lib/api";
import { Shell } from "./Shell";

/** `/`: the most recently updated run that has meetings. */
function LatestRun() {
  const { data, error, isLoading } = useRuns();
  if (isLoading) return <Spinner label="Loading runs" />;
  if (error) return <ErrorNote error={error} />;
  const runs = [...(data ?? [])].filter((r) => r.meetings > 0).sort((a, b) => b.updated - a.updated);
  if (!runs.length)
    return (
      <div className="p-8">
        <Empty>No runs found. Runs are read from SYNTHETICS_RUNS (default apps/Synthetics/runs).</Empty>
      </div>
    );
  return <Navigate to={`/runs/${encodeURIComponent(runs[0].id)}/meetings`} replace />;
}

function RunIndex() {
  const { run = "" } = useParams();
  return <Navigate to={`/runs/${encodeURIComponent(run)}/meetings`} replace />;
}

function NotFound() {
  return (
    <>
      <PageHeader title="Not found" />
      <Empty>There is no page here.</Empty>
    </>
  );
}

export const router = createBrowserRouter([
  { path: "/", element: <LatestRun /> },
  {
    path: "/runs/:run",
    element: <Shell />,
    children: [
      { index: true, element: <RunIndex /> },
      { path: "meetings", element: <MeetingBrowser /> },
      { path: "meetings/:meeting", element: <MeetingDetail /> },
      { path: "distributions", element: <Distributions /> },
      { path: "*", element: <NotFound /> },
    ],
  },
  { path: "*", element: <NotFound /> },
]);
