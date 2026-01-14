import React, { useState, useEffect } from "react";
import { ReactSortable } from "react-sortablejs";
import { useDispatch, useSelector } from "react-redux";
import ExperimentComponent from "./ExperimentComponent";
import GenericTabBar from "./GenericTabBar";

//import { FixedSizeList as List } from "react-window";

import * as experimentSlice from "../state/slices/ExperimentSlice";

import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner.js";

import { Shape } from "./WellSelectorCanvas.js";

import {
  Button,
  Input,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
} from "@mui/material";
//##################################################################################

export const ViewMode = Object.freeze({
  NAME: "name",
  POSITION: "position",
  SHAPE: "shape",
});

//##################################################################################
const PointListEditorComponent = () => {
  //local state
  const [viewMode, setViewMode] = useState(ViewMode.NAME);

  //redux dispatcher
  const dispatch = useDispatch();

  // Access global Redux state
  const experimentState = useSelector(experimentSlice.getExperimentState);
  //console.log("PointListEditorComponent", experimentState);
  //console.log("PointListEditorComponent", experimentState.pointList);

  //##################################################################################
  const setOrderedList = (newList) => {
    //console.log("setOrderedList", newList);
    // Clean the list by removing ReactSortable's internal properties before saving to Redux
    const cleanedList = newList.map(({ chosen, selected, filtered, ...point }) => point);
    dispatch(experimentSlice.setPointList(cleanedList)); // Update Redux state
  };

  //##################################################################################
  const handlePointChanged = (index, axis, value) => {
    // Handle input changes for x and y values
    const updatedPoints = experimentState.pointList.map((point, idx) => {
      if (idx === index) {
        return { ...point, [axis]: value };
      }
      return point;
    });


    // Update Redux state
    dispatch(experimentSlice.setPointList(updatedPoints));
  };

  //##################################################################################
  const handleRemovePoint = (index) => {
    // Update Redux state
    dispatch(experimentSlice.removePoint(index));
  };

  //##################################################################################
  const handleCreatePoint = (id) => {
    //create new point TODO change to e.g. PointBuilder
    const newPoint = { x: 100000, y: 100000 };
    // Update Redux state
    dispatch(experimentSlice.createPoint(newPoint));
  };

  //##################################################################################
  const handleDeleteAll = () => {
    // Update Redux state
    dispatch(experimentSlice.setPointList([]));
  };
  //##################################################################################
  const handleGotoButtonClick = (x, y) => {
    console.log("handleGotoButtonClick", x, y);
    // Do something with x and y
    apiPositionerControllerMovePositioner({
      axis: "X",
      dist: x,
      isAbsolute: true,
      speed: 20000,
    })
      .then((positionerResponse) => {
        console.log(
          "apiPositionerControllerMovePositioner X",
          positionerResponse
        );
      })
      .catch((error) => {
        console.error(
          "apiPositionerControllerMovePositioner X",
          "Error moving position:",
          error
        );
      });

    apiPositionerControllerMovePositioner({
      axis: "Y",
      dist: y,
      isAbsolute: true,
      speed: 20000,
    })
      .then((positionerResponse) => {
        console.log(
          "apiPositionerControllerMovePositioner Y",
          positionerResponse
        );
      })
      .catch((error) => {
        console.error(
          "apiPositionerControllerMovePositioner Y",
          "Error moving position:",
          error
        );
      });
  };

  //##################################################################################
  const PointItem = ({ item, index }) => {
    // Render the item (customize as needed)
    return (
      <div key={index}>
        <p>{item.name}</p> {/* or however you want to display the item */}
        {/* Other item details */}
      </div>
    );
  };

  //##################################################################################
  return (
    <div
      style={{
        textAlign: "left",
        padding: "10px",
        margin: "0px",
        //minWidth: "600px",
      }}
    >
      {/* Header */}
      <h4 style={{ margin: "0", padding: "0" }}>Point List Editor</h4>

      {/* Add             <ExperimentComponent /> */}
      <ExperimentComponent />
      <h1> </h1>

      {/* Remove item button */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          width: "100%",
          marginBottom: "8px",
        }}
      >
        {/* create point*/}
        <Button variant="contained" onClick={handleCreatePoint}>
          +
        </Button>

        {/* view mode */}
        <FormControl>
          <InputLabel>Parameter</InputLabel>
          <Select
            sx={{ height: "32px" }}
            label="Parameter"
            value={viewMode}
            onChange={(e) => setViewMode(e.target.value)}
          >
            <MenuItem value={ViewMode.NAME}>Name</MenuItem>
            <MenuItem value={ViewMode.POSITION}>Position</MenuItem>
            <MenuItem value={ViewMode.SHAPE}>Shape</MenuItem>
          </Select>
        </FormControl>

        {/* delete all */}
        <Button variant="contained" onClick={handleDeleteAll}>
          delete all
        </Button>
      </div>

      {/* Sortable list */}
      <ReactSortable
        filter=".addImageButtonContainer"
        dragClass="sortableDrag"
        list={experimentState.pointList.map(point => ({ ...point }))} // Create mutable copies for ReactSortable
        setList={setOrderedList}
        animation="200"
        easing="ease-out"
        //onEnd={handleOnDragEnd}
        handle=".drag-handle" // Only the drag icon (handle) will be draggable
      >
        {experimentState.pointList.map(
          (
            item,
            index //for each point
          ) => (
            <div //<------------------------------------------BEGIN ITEM
              key={item.id} //ReactSortable item key
              className="draggableItem"
              style={{
                textAlign: "left",
                padding: "4px",
                margin: "0px",
                backgroundColor: "",
                display: "flex", // Enable Flexbox for horizontal alignment
                flexDirection: "row", // Arrange children in a row
                gap: "4px", // Optional: space between items
                fontSize: "18px",
                marginTop: "-1px",
                //flexWrap: "wrap", // Allow the items to wrap if the screen is small
              }}
            >
              {/* Point INDEX */}
              <div
                className="drag-handle"
                style={{
                  cursor: "move", // Change cursor to indicate draggable area // {item.name}
                  width: "32px",
                }}
              >
                ≡{index}:
              </div>

              {/* Input for name value */}
              {viewMode == ViewMode.NAME && <div></div>}
              {viewMode == ViewMode.NAME && (
                <Input
                  type="text"
                  value={item.name}
                  onChange={(e) =>
                    handlePointChanged(index, "name", e.target.value)
                  }
                  placeholder="name"
                  style={{ flex: 1 }} // Ensures inputs take equal space
                />
              )}

              {/* Input for x value */}
              {viewMode == ViewMode.POSITION && <div>x:</div>}
              {viewMode == ViewMode.POSITION && (
                <Input
                  type="number"
                  value={item.x}
                  onChange={(e) =>
                    handlePointChanged(index, "x", parseFloat(e.target.value))
                  }
                  placeholder="X value"
                  style={{ flex: 1 }} // Ensures inputs take equal space
                />
              )}

              {/* Input for y value */}
              {viewMode == ViewMode.POSITION && <div>y:</div>}
              {viewMode == ViewMode.POSITION && (
                <Input
                  type="number"
                  value={item.y}
                  onChange={(e) =>
                    handlePointChanged(index, "y", parseFloat(e.target.value))
                  }
                  placeholder="Y value"
                  style={{ flex: 1 }} // Ensures inputs take equal space
                />
              )}

              {/* Input for shape value */}
              {viewMode == ViewMode.SHAPE && (
                <>
                  <div>S</div>
                  <FormControl>
                    <Select
                      sx={{ height: "32px" }}
                      value={item.shape}
                      onChange={(e) =>
                        handlePointChanged(index, "shape", e.target.value)
                      }
                    >
                      <MenuItem value={Shape.EMPTY}>Off</MenuItem>
                      <MenuItem value={Shape.RECTANGLE}>Rect</MenuItem>
                      <MenuItem value={Shape.CIRCLE}>Circle</MenuItem>
                    </Select>
                  </FormControl>
                </>
              )}

              {/* Input for rect value */}
              {viewMode == ViewMode.SHAPE && item.shape == Shape.RECTANGLE && (
                <>
                  <div>±X:</div>
                  <Input
                    type="number"
                    value={item.rectMinusX}
                    min="-1000000"
                    max="0"
                    onChange={(e) =>{
                      handlePointChanged(
                        index,
                        "rectMinusX",
                        parseInt(e.target.value, 10)
                      )
                    }}
                    placeholder="Y value"
                    style={{ flex: 1 }} // Ensures inputs take equal space
                  />
                  <div></div>
                  <Input
                    type="number"
                    value={item.rectPlusX}
                    min="0"
                    max="1000000"
                    onChange={(e) => {
                      handlePointChanged(
                        index,
                        "rectPlusX",
                        parseInt(e.target.value, 10)
                      );
                    }}
                    placeholder="Y value"
                    style={{ flex: 1 }} // Ensures inputs take equal space
                  />
                  <div>±Y:</div>
                  <Input
                    type="number"
                    value={item.rectMinusY}
                    min="-1000000"
                    max="0"
                    onChange={(e) => {
                      handlePointChanged(
                        index,
                        "rectMinusY",
                        parseInt(e.target.value, 10)
                      )
                    }}
                    placeholder="Y value"
                    style={{ flex: 1 }} // Ensures inputs take equal space
                  />
                  <div></div>
                  <Input
                    type="number"
                    value={item.rectPlusY}
                    min="0"
                    max="1000000"
                    onChange={(e) =>{
                      handlePointChanged(
                        index,
                        "rectPlusY",
                        parseInt(e.target.value, 10)
                      )
                    }}
                    placeholder="Y value"
                    style={{ flex: 1 }} // Ensures inputs take equal space
                  />
                </>
              )}

              
              {/* Input for circle value */}
              {viewMode == ViewMode.SHAPE && item.shape == Shape.CIRCLE && (
                <>
                  <div>rX:</div>
                  <Input
                    type="number"
                    value={item.circleRadiusX}
                    min="-1000000"
                    max="0"
                    onChange={(e) =>{
                      handlePointChanged(
                        index,
                        "circleRadiusX",
                        parseInt(e.target.value, 10)
                      )
                    }}
                    placeholder="Y value"
                    style={{ flex: 1 }} // Ensures inputs take equal space
                  />
                  <div>rY:</div>
                  <Input
                    type="number"
                    value={item.circleRadiusY}
                    min="-1000000"
                    max="0"
                    onChange={(e) => {
                      handlePointChanged(
                        index,
                        "circleRadiusY",
                        parseInt(e.target.value, 10)
                      )
                    }}
                    placeholder="Y value"
                    style={{ flex: 1 }} // Ensures inputs take equal space
                  />
                </>
              )}

              {/* dummy button*/}
              {viewMode != ViewMode.READONLY && (
                <Button
                  sx={{ padding: "0px" }}
                  disabled={false}
                  onClick={() => handleGotoButtonClick(item.x, item.y)}
                >
                  Goto
                </Button>
              )}
              {/* Remove item button*/}
              {viewMode != ViewMode.READONLY && (
                <Button
                  sx={{ padding: "0px" }}
                  onClick={() => handleRemovePoint(index)}
                >
                  Delete
                </Button>
              )}
            </div> //<------------------------------------------END ITEM
          )
        )}
      </ReactSortable>
    </div>
  );
};

//##################################################################################
export default PointListEditorComponent;
