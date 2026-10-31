import type {Observation} from './contracts';
const bones=[['left_shoulder','right_shoulder'],['left_shoulder','left_elbow'],['left_elbow','left_wrist'],['right_shoulder','right_elbow'],['right_elbow','right_wrist'],['left_shoulder','left_hip'],['right_shoulder','right_hip'],['left_hip','right_hip'],['left_hip','left_knee'],['left_knee','left_ankle'],['right_hip','right_knee'],['right_knee','right_ankle'],['left_eye','right_eye'],['left_eye','left_ear'],['right_eye','right_ear']];
export function drawPose(c:CanvasRenderingContext2D,p:Observation,scale=1){
 const points=new Map((p.keypoints??[]).filter(k=>k.state==='detected'&&k.xy_px).map(k=>[k.name,k.xy_px!]));
 c.setLineDash([]);c.strokeStyle='#D8EF45';c.lineWidth=2*scale;
 for(const [a,b] of bones){const x=points.get(a),y=points.get(b);if(x&&y){c.beginPath();c.moveTo(...x);c.lineTo(...y);c.stroke();}}
 for(const xy of points.values()){c.beginPath();c.arc(...xy,4*scale,0,2*Math.PI);c.fillStyle='#46e650';c.fill();c.strokeStyle='#f3f0e6';c.lineWidth=scale;c.stroke();}
}
