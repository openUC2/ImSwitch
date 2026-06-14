import { useEffect, useState } from "react";
import { useSelection } from "../../contexts/SelectionContext";
import { getFileManagerBaseUrl } from "../../api/api";
import { getDataSize } from "../../utils/getDataSize";
import { formatDate } from "../../utils/formatDate";
import { FaRegFile, FaRegFolderOpen } from "react-icons/fa6";
import { MdClose } from "react-icons/md";
import "./MetadataPane.scss";

const LABEL_MAP = {
  format: "Format",
  shape: "Shape",
  axes: "Axes",
  dtype: "Data type",
  pixelType: "Pixel type",
  pixelSizeUm: "Pixel size",
  dimensions: "Dimensions",
  channels: "Channels",
  acquisitionDate: "Acquired",
  exposure: "Exposure",
  stagePosition: "Stage position",
  description: "Description",
  sizeBytes: "Size on disk",
  metadataError: "Warning",
};

const formatPixelSize = (ps) => {
  if (!ps) return null;
  const parts = [];
  for (const axis of ["x", "y", "z"]) {
    if (ps[axis] != null) parts.push(`${axis.toUpperCase()}: ${ps[axis]} µm`);
  }
  return parts.join(", ");
};

const formatDimensions = (dims) => {
  if (!dims) return null;
  return Object.entries(dims)
    .map(([k, v]) => `${k}: ${v}`)
    .join(", ");
};

const formatExposure = (exp) => {
  if (!exp) return null;
  return `${exp.value} ${exp.unit}`;
};

const formatStagePosition = (pos) => {
  if (!pos) return null;
  return Object.entries(pos)
    .map(([k, v]) => `${k.toUpperCase()}: ${Number(v).toFixed(1)}`)
    .join(", ");
};

const formatValue = (key, value) => {
  if (value == null) return null;
  switch (key) {
    case "pixelSizeUm":
      return formatPixelSize(value);
    case "dimensions":
      return formatDimensions(value);
    case "exposure":
      return formatExposure(value);
    case "stagePosition":
      return formatStagePosition(value);
    case "sizeBytes":
      return value > 0 ? getDataSize(value) : null;
    case "shape":
      return Array.isArray(value) ? value.join(" × ") : String(value);
    case "channels":
      return Array.isArray(value) ? value.join(", ") : String(value);
    default:
      return String(value);
  }
};

const MetadataPane = ({ onClose }) => {
  const { selectedFiles } = useSelection();
  const [metadata, setMetadata] = useState(null);
  const [loading, setLoading] = useState(false);
  const [thumbError, setThumbError] = useState(false);

  const file = selectedFiles.length === 1 ? selectedFiles[0] : null;

  useEffect(() => {
    if (!file) {
      setMetadata(null);
      return;
    }
    setThumbError(false);

    if (!file.isImage) {
      setMetadata(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
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
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [file?.path]);

  if (!file) {
    return (
      <div className="metadata-pane empty">
        <p>Select a file to view details</p>
      </div>
    );
  }

  const thumbUrl =
    file.thumbnailPath && !thumbError
      ? `${getFileManagerBaseUrl()}${file.thumbnailPath}?size=512`
      : null;

  const displayKeys = Object.keys(LABEL_MAP).filter(
    (k) => metadata && metadata[k] != null && k !== "name" && k !== "isImage" && k !== "imageType"
  );

  return (
    <div className="metadata-pane">
      <div className="metadata-header">
        <span className="metadata-title text-truncate" title={file.name}>
          {file.name}
        </span>
        <button className="metadata-close" onClick={onClose}>
          <MdClose size={18} />
        </button>
      </div>

      <div className="metadata-body">
        {thumbUrl ? (
          <div className="metadata-thumbnail">
            <img
              src={thumbUrl}
              alt={file.name}
              onError={() => setThumbError(true)}
            />
          </div>
        ) : (
          <div className="metadata-icon">
            {file.isDirectory ? (
              <FaRegFolderOpen size={64} />
            ) : (
              <FaRegFile size={64} />
            )}
          </div>
        )}

        <div className="metadata-info-basic">
          {file.size > 0 && (
            <span className="metadata-size">{getDataSize(file.size)}</span>
          )}
          {(file.modifiedTime || file.updatedAt) && (
            <span className="metadata-date">
              {formatDate(file.modifiedTime || file.updatedAt)}
            </span>
          )}
          {file.imageType && (
            <span className="metadata-badge">{file.imageType.toUpperCase()}</span>
          )}
        </div>

        {loading && <div className="metadata-loading">Loading metadata...</div>}

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

        {metadata && !loading && displayKeys.length === 0 && (
          <div className="metadata-empty">No metadata available</div>
        )}
      </div>
    </div>
  );
};

export default MetadataPane;
