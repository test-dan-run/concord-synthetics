import { createBrowserRouter, Navigate, useParams } from "react-router-dom";
import { Empty, ErrorNote, PageHeader, Spinner } from "../components/ui";
import { SpeakerPool } from "../features/admin/SpeakerPool";
import { NewDataset } from "../features/builder/NewDataset";
import { MeetingBrowser } from "../features/browser/MeetingBrowser";
import { MeetingDetail } from "../features/detail/MeetingDetail";
import { Distributions } from "../features/distributions/Distributions";
import { Describe } from "../features/guided/Describe";
import { Export } from "../features/guided/Export";
import { GenerateReview } from "../features/guided/GenerateReview";
import { Pilot } from "../features/guided/Pilot";
import { Plan } from "../features/guided/Plan";
import { Pipeline } from "../features/pipeline/Pipeline";
import { useRuns, useViewer } from "../lib/api";
import { Shell } from "./Shell";

/** `/`: the guided flow when the viewer can create runs, else the most recently updated run that has meetings. */
function Home() {
  const viewer = useViewer();
  const { data, error, isLoading } = useRuns();
  if (isLoading || viewer.isLoading) return <Spinner label="Loading runs" />;
  if (error) return <ErrorNote error={error} />;
  if (viewer.data && !viewer.data.read_only) return <Navigate to="/create" replace />;
  const runs = [...(data ?? [])].filter((r) => r.meetings > 0).sort((a, b) => b.updated - a.updated);
  if (!runs.length) return <Navigate to="/new" replace />;
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
  { path: "/", element: <Home /> },
  { path: "/new", element: <Shell />, children: [{ index: true, element: <NewDataset /> }] },
  {
    path: "/create",
    element: <Shell />,
    children: [
      { index: true, element: <Describe /> },
      { path: "plan", element: <Plan /> },
      { path: "pilot/:run", element: <Pilot /> },
    ],
  },
  { path: "/admin", element: <Shell />, children: [{ path: "speakers", element: <SpeakerPool /> }] },
  {
    path: "/runs/:run",
    element: <Shell />,
    children: [
      { index: true, element: <RunIndex /> },
      { path: "meetings", element: <MeetingBrowser /> },
      { path: "meetings/:meeting", element: <MeetingDetail /> },
      { path: "distributions", element: <Distributions /> },
      { path: "pipeline", element: <Pipeline /> },
      { path: "review", element: <GenerateReview /> },
      { path: "export", element: <Export /> },
      { path: "*", element: <NotFound /> },
    ],
  },
  { path: "*", element: <NotFound /> },
]);
