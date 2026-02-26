export const formatDate = (date) => {
  if (!date) return "";

  let parsedDate = date;
  if (typeof date === "number") {
    // Backend sends unix timestamp in seconds
    const millis = date < 1e12 ? date * 1000 : date;
    parsedDate = new Date(millis);
  } else {
    const parsed = Date.parse(date);
    if (isNaN(parsed)) return "";
    parsedDate = new Date(parsed);
  }

  const dateObj = parsedDate;
  let hours = dateObj.getHours();
  const minutes = dateObj.getMinutes();
  const ampm = hours >= 12 ? "PM" : "AM";
  hours = hours % 12;
  hours = hours ? hours : 12;
  const month = dateObj.getMonth() + 1;
  const day = dateObj.getDate();
  const year = dateObj.getFullYear();

  return `${month}/${day}/${year} ${hours}:${minutes < 10 ? "0" + minutes : minutes} ${ampm}`;
};
