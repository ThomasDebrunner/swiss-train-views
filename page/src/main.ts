import './style.css'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'
import proj4 from 'proj4'

import data from '../train_data.json'

const lv03 = "+proj=somerc +lat_0=46.9524055555556 +lon_0=7.43958333333333 +k_0=1 +x_0=600000 +y_0=200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs +type=crs";
const wgs84 = proj4.WGS84;

function lv03ToWGS84(x: number, y: number) {
  const res = proj4(lv03, wgs84, [x, y]);
  return [res[1], res[0]];
}


function plotLine(line: any, map: any) {
  const points = line['points']
  const vectors = line['vectors']
  const left_distances = line['left_distances']
  const right_distances = line['right_distances']

  const leftPoints = points.map((p: any, i: number) => {
    const vector = vectors[i]
    const left_distance = left_distances[i]
    const left_vector = [vector[1], -vector[0]]
    return [p[0] + left_vector[0] * left_distance, p[1] + left_vector[1] * left_distance]
  })

  const rightPoints = points.map((p: any, i: number) => {
    const vector = vectors[i]
    const right_distance = right_distances[i]
    const right_vector = [-vector[1], vector[0]]
    return [p[0] + right_vector[0] * right_distance, p[1] + right_vector[1] * right_distance]
  })


  const points_wgs = points.map((p: any) => lv03ToWGS84(p[0], p[1]))
  const leftPoints_wgs = leftPoints.map((p: any) => lv03ToWGS84(p[0], p[1]))
  const rightPoints_wgs = rightPoints.map((p: any) => lv03ToWGS84(p[0], p[1]))
  leftPoints_wgs.reverse()
  rightPoints_wgs.reverse()

  // draw polyline
  L.polygon([...points_wgs, ...leftPoints_wgs], { color: 'purple' }).addTo(map);
  L.polygon([...points_wgs, ...rightPoints_wgs], { color: 'purple' }).addTo(map);
  L.polyline(points_wgs, { color: 'red', weight: 3 }).addTo(map);
}

async function main() {
  const map = L.map('map').setView([46.9572, 8.3660], 9);

  // load a tile layer
  L.tileLayer('https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-farbe/default/current/3857/{z}/{x}/{y}.jpeg',
    {
      attribution: 'Data: &copy; <a href="https://www.swisstopo.admin.ch/en/home.html">swisstopo</a>',
      maxZoom: 17,
      minZoom: 8
    }).addTo(map);

  // plot lines
  data.forEach((line: any) => {
    plotLine(line, map)
  })



}


void main()