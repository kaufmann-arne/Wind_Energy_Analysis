// src/data/heatmapData.ts
import raw from "./windscore_points.json";

export type ClassifiedPoint = {
  lat: number;
  lng: number;
  classId: 0 | 1 | 2 | 3 | 4;
};

type RawPoint = [number, number, 0 | 1 | 2 | 3 | 4];

function toPoints(data: RawPoint[]): ClassifiedPoint[] {
  return data.map(([lat, lng, classId]) => ({ lat, lng, classId }));
}

export const heatmapData: ClassifiedPoint[] = toPoints(raw as RawPoint[]);
