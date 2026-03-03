import { createContext, useContext, useEffect, useRef, useState } from "react";
import { useFiles } from "./FilesContext";

const FileNavigationContext = createContext();

export const FileNavigationProvider = ({ children, initialPath }) => {
  const { files } = useFiles();
  const isMountRef = useRef(false);
  const lastAppliedInitialPathRef = useRef(null);
  const [currentPath, setCurrentPath] = useState("");
  const [currentFolder, setCurrentFolder] = useState(null);
  const [currentPathFiles, setCurrentPathFiles] = useState([]);

  useEffect(() => {
    if (Array.isArray(files) && files.length > 0) {
      setCurrentPathFiles(() => {
        return files.filter(
          (file) => file.path === `${currentPath}/${file.name}`,
        );
      });

      setCurrentFolder(() => {
        return files.find((file) => file.path === currentPath) ?? null;
      });
    }
  }, [files, currentPath]);

  useEffect(() => {
    if (!isMountRef.current && Array.isArray(files) && files.length > 0) {
      const nextPath = files.some((file) => file.path === initialPath)
        ? initialPath
        : "";
      setCurrentPath(nextPath);
      lastAppliedInitialPathRef.current = nextPath;
      isMountRef.current = true;
    }
  }, [initialPath, files]);

  // Handle changes to initialPath after initial mount
  useEffect(() => {
    if (
      isMountRef.current &&
      Array.isArray(files) &&
      files.length > 0 &&
      initialPath
    ) {
      if (lastAppliedInitialPathRef.current === initialPath) {
        return;
      }
      const pathExists = files.some((file) => file.path === initialPath);
      if (pathExists) {
        setCurrentPath(initialPath);
        lastAppliedInitialPathRef.current = initialPath;
      }
    }
  }, [initialPath, files]);

  return (
    <FileNavigationContext.Provider
      value={{
        currentPath,
        setCurrentPath,
        currentFolder,
        setCurrentFolder,
        currentPathFiles,
        setCurrentPathFiles,
      }}
    >
      {children}
    </FileNavigationContext.Provider>
  );
};

export const useFileNavigation = () => useContext(FileNavigationContext);
