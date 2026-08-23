import { Navigate, Route, Routes } from "react-router-dom";
import { Dashboard } from "@/routes/Dashboard";
import { Walkthrough } from "@/walkthrough/Walkthrough";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Walkthrough />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;
