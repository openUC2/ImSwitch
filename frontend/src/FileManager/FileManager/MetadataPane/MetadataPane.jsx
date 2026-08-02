import { useEffect, useState } from "react";
import { useSelection } from "../../contexts/SelectionContext";
import { getFileManagerBaseUrl } from "../../api/api";
import { getDataSize } from "../../utils/getDataSize";
import { formatDate } from "../../utils/formatDate";
import {
  LABEL_MAP,
  formatValue,
  formatSimpleValue,
  groupAcquisitionAttributes,
} from "../../utils/metadataFormat";
import { FaRegFile, FaRegFolderOpen } from "react-icons/fa6";
import { MdClose, MdExpandMore, MdChevronRight } from "react-icons/md";
import "./MetadataPane.scss";

const CollapsibleSection = ({ title, children, defaultOpen = false }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="collapsible-section">
      <button
        className="collapsible-header"
        onClick={() => setOpen((v) => !v)}
      >
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
                          <td className="metadata-value">
                            {formatSimpleValue(v)}
                          </td>
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
          <>
            <table className="metadata-table">
              <tbody>
                {displayKeys
                  .filter((k) => k !== "description")
                  .map((key) => {
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
            {metadata.description && (
              <DescriptionBlock value={metadata.description} />
            )}
          </>
        )}

        {metadata && !loading && displayKeys.length === 0 && (
          <div className="metadata-empty">No metadata available</div>
        )}
      </div>
    </div>
  );
};

export default MetadataPane;
