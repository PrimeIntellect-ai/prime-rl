import React from "react";
import ReactDOM from "react-dom/client";
import { Theme } from "@radix-ui/themes";
import { App } from "./App";
import "@radix-ui/themes/styles.css";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Theme appearance="dark" accentColor="gray" grayColor="slate" radius="large" scaling="100%">
      <App />
    </Theme>
  </React.StrictMode>,
);
