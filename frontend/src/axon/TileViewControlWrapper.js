
import TileViewComponent from "./TileViewComponent";
import PositionControllerComponent from "./PositionControllerComponent";


 
 
 
const TileViewControlWrapper = () => {
  return ( 
    <div style={{ position: "relative", width: "400", height: "300" }}>
        <div style={{ position: "relative", top: "0px", left: "0px", zIndex: 1, }}>
        <TileViewComponent />
        </div>
        <div style={{ position: "absolute", top: "0px", left: "0px", zIndex: 2, }}>
        <PositionControllerComponent />
        </div>
    </div>
  );
}

export default TileViewControlWrapper;

