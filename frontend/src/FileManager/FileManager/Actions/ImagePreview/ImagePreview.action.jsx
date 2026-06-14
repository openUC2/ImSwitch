import { useEffect, useState } from "react";
import { useSelection } from "../../../contexts/SelectionContext";
import { getFileManagerBaseUrl } from "../../../api/api";
import { getDataSize } from "../../../utils/getDataSize";
import {
  LABEL_MAP,
  formatValue,
  formatSimpleValue,
  groupAcquisitionAttributes,
} from "../../../utils/metadataFormat";
import { MdOutlineFileDownload, MdOpenInNew, MdExpandMore, MdChevronRight } from "react-icons/md";
import { FaRegFile } from "react-icons/fa6";
import { SiImagej } from "react-icons/si";
import { MdViewInAr } from "react-icons/md";
import Button from "../../../components/Button/Button";
import Loader from "../../../components/Loader/Loader";
import "./ImagePreview.action.scss";

const CollapsibleSection = ({ title, children, defaultOpen = false }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="collapsible-section">
      <button className="collapsible-header" onClick={() => setOpen((v) => !v)}>
        {open ? <MdExpandMore size={16} /> : <MdChevronRight size={16} />}
        <span>{title}</span>
      </button>
      {open && <div className="collapsible-body">{children}</div>}
    </div>
  );
};

const DescriptionBlock = ({ value }) => {
  let parsed = value;
  if (typeof value === "string") {
    try {
      parsed = JSON.parse(value);
    } catch {
      return <span className="metadata-value">{value}</span>;
    }
  }
  if (typeof parsed !== "object" || parsed === null) {
    return <span className="metadata-value">{String(parsed)}</span>;
  }

  const topLevel = {};
  const nested = {};
  for (const [k, v] of Object.entries(parsed)) {
    if (typeof v === "object" && v !== null && !Array.isArray(v)) {
      nested[k] = v;
    } else {
      topLevel[k] = v;
    }
  }

  return (
    <div className="description-block">
      {Object.keys(topLevel).length > 0 && (
        <table className="metadata-table">
          <tbody>
            {Object.entries(topLevel).map(([k, v]) => (
              <tr key={k}>
                <td className="metadata-label">{k}</td>
                <td className="metadata-value">{formatSimpleValue(v)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {Object.entries(nested).map(([sectionName, sectionData]) => {
        if (sectionName === "AcquisitionAttributes") {
          const groups = groupAcquisitionAttributes(sectionData);
          return (
            <CollapsibleSection key={sectionName} title={sectionName}>
              {Object.entries(groups).map(([groupName, entries]) => (
                <CollapsibleSection key={groupName} title={groupName}>
                  <table className="metadata-table compact">
                    <tbody>
                      {entries.map(({ label, value: v }) => (
                        <tr key={label}>
                          <td className="metadata-label">{label}</td>
                          <td className="metadata-value">{formatSimpleValue(v)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </CollapsibleSection>
              ))}
            </CollapsibleSection>
          );
        }
        return (
          <CollapsibleSection key={sectionName} title={sectionName}>
            <table className="metadata-table compact">
              <tbody>
                {Object.entries(sectionData).map(([k, v]) => (
                  <tr key={k}>
                    <td className="metadata-label">{k}</td>
                    <td className="metadata-value">{formatSimpleValue(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CollapsibleSection>
        );
      })}
    </div>
  );
};

const ImagePreviewAction = ({ onOpenWithVizarr, triggerAction }) => {
  const { selectedFiles } = useSelection();
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const [metadata, setMetadata] = useState(null);
  const [metaLoading, setMetaLoading] = useState(false);

  const file = selectedFiles[0];

  useEffect(() => {
    if (!file) return;
    let cancelled = false;
    setMetaLoading(true);
    const url = `${getFileManagerBaseUrl()}/metadata${file.path}`;
    fetch(url)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
      .then((data) => {
        if (!cancelled) setMetadata(data);
      })
      .catch(() => {
        if (!cancelled) setMetadata(null);
      })
      .finally(() => {
        if (!cancelled) setMetaLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [file?.path]);

  if (!file) return null;

  const base = getFileManagerBaseUrl();
  const thumbUrl = file.thumbnailPath
    ? `${base}${file.thumbnailPath}?size=1024`
    : null;
  const cleanPath = file.path.startsWith("/") ? file.path.slice(1) : file.path;
  const downloadUrl = `${base}/download/${cleanPath}`;

  const isZarr =
    file.name?.toLowerCase().endsWith(".zarr") ||
    file.name?.toLowerCase().endsWith(".ome.zarr");

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

  const handleOpenInVizarr = () => {
    if (onOpenWithVizarr) {
      onOpenWithVizarr(file);
      triggerAction.close();
    }
  };

  const displayKeys = metadata
    ? Object.keys(LABEL_MAP).filter(
        (k) =>
          metadata[k] != null &&
          k !== "name" &&
          k !== "isImage" &&
          k !== "imageType" &&
          k !== "description"
      )
    : [];

  return (
    <section className="image-preview-action">
      <div className="preview-layout">
        <div className="preview-left">
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
            <Button onClick={handleDownload} padding="0.4rem 0.7rem">
              <div className="action-btn">
                <MdOutlineFileDownload size={16} />
                <span>Download</span>
              </div>
            </Button>
            <Button onClick={handleOpenInTab} padding="0.4rem 0.7rem">
              <div className="action-btn">
                <MdOpenInNew size={16} />
                <span>Open in tab</span>
              </div>
            </Button>
            <Button onClick={handleOpenInFiji} padding="0.4rem 0.7rem">
              <div className="action-btn">
                <SiImagej size={16} />
                <span>Fiji</span>
              </div>
            </Button>
            {isZarr && onOpenWithVizarr && (
              <Button onClick={handleOpenInVizarr} padding="0.4rem 0.7rem" type="secondary">
                <div className="action-btn">
                  <MdViewInAr size={16} />
                  <span>OME-Zarr Viewer</span>
                </div>
              </Button>
            )}
          </div>
        </div>

        <div className="preview-right">
          <div className="preview-meta-header">
            <span className="preview-filename text-truncate" title={file.name}>
              {file.name}
            </span>
            {file.imageType && (
              <span className="preview-badge">{file.imageType.toUpperCase()}</span>
            )}
          </div>

          {file.size > 0 && (
            <span className="preview-filesize">{getDataSize(file.size)}</span>
          )}

          {metaLoading && <span className="preview-meta-loading">Loading metadata...</span>}

          {metadata && displayKeys.length > 0 && (
            <table className="metadata-table">
              <tbody>
                {displayKeys.map((key) => {
                  const formatted = formatValue(key, metadata[key]);
                  if (formatted == null) return null;
                  return (
                    <tr key={key}>
                      <td className="metadata-label">{LABEL_MAP[key]}</td>
                      <td className="metadata-value">{formatted}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}

          {metadata?.description && (
            <DescriptionBlock value={metadata.description} />
          )}
        </div>
      </div>
    </section>
  );
};

export default ImagePreviewAction;
