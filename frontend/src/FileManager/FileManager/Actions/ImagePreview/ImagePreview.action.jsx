import { useState } from "react";
import { useSelection } from "../../../contexts/SelectionContext";
import { getFileManagerBaseUrl } from "../../../api/api";
import { MdOutlineFileDownload, MdOpenInNew } from "react-icons/md";
import { FaRegFile } from "react-icons/fa6";
import { SiImagej } from "react-icons/si";
import Button from "../../../components/Button/Button";
import Loader from "../../../components/Loader/Loader";
import "./ImagePreview.action.scss";

const ImagePreviewAction = () => {
  const { selectedFiles } = useSelection();
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  const file = selectedFiles[0];
  if (!file) return null;

  const base = getFileManagerBaseUrl();
  const thumbUrl = file.thumbnailPath
    ? `${base}${file.thumbnailPath}?size=1024`
    : null;
  const cleanPath = file.path.startsWith("/") ? file.path.slice(1) : file.path;
  const downloadUrl = `${base}/download/${cleanPath}`;

  const handleDownload = () => {
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.setAttribute("download", file.name);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleOpenInTab = () => {
    window.open(downloadUrl, "_blank");
  };

  const handleOpenInFiji = () => {
    const fijiUrl = `fiji://open?url=${encodeURIComponent(downloadUrl)}`;
    window.open(fijiUrl, "_self");
  };

  return (
    <section className="image-preview-action">
      <div className="preview-container">
        {thumbUrl && !hasError ? (
          <>
            <Loader isLoading={isLoading} />
            <img
              src={thumbUrl}
              alt={file.name}
              className={`preview-image ${isLoading ? "img-loading" : ""}`}
              onLoad={() => {
                setIsLoading(false);
                setHasError(false);
              }}
              onError={() => {
                setIsLoading(false);
                setHasError(true);
              }}
            />
          </>
        ) : (
          <div className="preview-fallback">
            <FaRegFile size={64} />
            <span>Preview not available</span>
          </div>
        )}
      </div>

      <div className="preview-actions">
        <Button onClick={handleDownload} padding="0.45rem 0.9rem">
          <div className="action-btn">
            <MdOutlineFileDownload size={18} />
            <span>Download</span>
          </div>
        </Button>
        <Button onClick={handleOpenInTab} padding="0.45rem 0.9rem">
          <div className="action-btn">
            <MdOpenInNew size={18} />
            <span>Open in new tab</span>
          </div>
        </Button>
        <Button onClick={handleOpenInFiji} padding="0.45rem 0.9rem">
          <div className="action-btn">
            <SiImagej size={18} />
            <span>Open in Fiji</span>
          </div>
        </Button>
      </div>
    </section>
  );
};

export default ImagePreviewAction;
