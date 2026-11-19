/**
 * Laser Settings Controller Component
 *
 * Shows the firmware PWM parameters of every laser (current value, max value,
 * resolution, frequency) next to the ImSwitch value range, and lets the user
 * change PWM frequency / resolution on the device.
 */

import React, { useState, useEffect, useCallback } from "react";
import {
  Box,
  Typography,
  Button,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  TextField,
  Alert,
} from "@mui/material";
import apiLaserControllerGetLaserNames from "../backendapi/apiLaserControllerGetLaserNames.js";
import apiLaserControllerGetLaserValueRanges from "../backendapi/apiLaserControllerGetLaserValueRanges.js";
import apiLaserControllerGetLaserPWMSettings from "../backendapi/apiLaserControllerGetLaserPWMSettings.js";
import apiLaserControllerSetLaserPWMSettings from "../backendapi/apiLaserControllerSetLaserPWMSettings.js";

const LaserSettingsController = () => {
  const [lasers, setLasers] = useState([]); // [{name, range, pwm, freq, res}]
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const names = await apiLaserControllerGetLaserNames();
      const rows = await Promise.all(
        names.map(async (name) => {
          const [range, pwm] = await Promise.all([
            apiLaserControllerGetLaserValueRanges(name).catch(() => null),
            apiLaserControllerGetLaserPWMSettings(name).catch(() => null),
          ]);
          return { name, range, pwm, freq: pwm?.LASERFreq ?? "", res: pwm?.LASERRes ?? "" };
        })
      );
      setLasers(rows);
      setError(null);
    } catch (e) {
      setError(`Could not load lasers: ${e.message}`);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const edit = (name, key, value) =>
    setLasers((ls) => ls.map((l) => (l.name === name ? { ...l, [key]: value } : l)));

  const apply = async (l) => {
    try {
      await apiLaserControllerSetLaserPWMSettings(l.name, Number(l.freq), Number(l.res));
      await load();
    } catch (e) {
      setError(`Could not set PWM for ${l.name}: ${e.message}`);
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>
        Laser Settings
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        PWM parameters reported by the firmware. Max value = 2^resolution (fully on). Applying
        new PWM settings switches the laser off; CAN lasers are read-only.
      </Typography>
      {error && <Alert severity="error">{error}</Alert>}
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Laser</TableCell>
            <TableCell>ID</TableCell>
            <TableCell>ImSwitch range</TableCell>
            <TableCell>Value</TableCell>
            <TableCell>Max</TableCell>
            <TableCell>Resolution (bit)</TableCell>
            <TableCell>Frequency (Hz)</TableCell>
            <TableCell />
          </TableRow>
        </TableHead>
        <TableBody>
          {lasers.map((l) => {
            const remote = l.pwm?.node !== undefined;
            return (
              <TableRow key={l.name}>
                <TableCell>{l.name}</TableCell>
                <TableCell>
                  {l.pwm?.LASERid ?? "–"}
                  {remote && ` (CAN ${l.pwm.node})`}
                </TableCell>
                <TableCell>{l.range ? `${l.range[0]} – ${l.range[1]}` : "–"}</TableCell>
                {l.pwm && !l.pwm.error ? (
                  <>
                    <TableCell>{l.pwm.LASERval}</TableCell>
                    <TableCell>{l.pwm.LASERmax}</TableCell>
                    <TableCell>
                      <TextField
                        size="small"
                        type="number"
                        value={l.res}
                        disabled={remote}
                        inputProps={{ min: 1, max: 16 }}
                        onChange={(e) => edit(l.name, "res", e.target.value)}
                      />
                    </TableCell>
                    <TableCell>
                      <TextField
                        size="small"
                        type="number"
                        value={l.freq}
                        disabled={remote}
                        onChange={(e) => edit(l.name, "freq", e.target.value)}
                      />
                    </TableCell>
                    <TableCell>
                      {!remote && (
                        <Button size="small" variant="outlined" onClick={() => apply(l)}>
                          Apply
                        </Button>
                      )}
                    </TableCell>
                  </>
                ) : (
                  <TableCell colSpan={5}>
                    {l.pwm?.error || "Not reported (non-ESP32 laser or old firmware/uc2rest)"}
                  </TableCell>
                )}
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
      <Button sx={{ mt: 2 }} variant="contained" onClick={load}>
        Refresh
      </Button>
    </Box>
  );
};

export default LaserSettingsController;
