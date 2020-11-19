# Mobile phone video - living room - frame skip _k=3_

The videos below show RidgeSfM reconstructions for a video captured on a mobile phone (every 3rd frame).

For each scene, we use the reconstructed depth and camera parameters to reproject the pixels to form a point cloud.
Each point in the cloud has the form _(x,y,z,r,g,b)_ ∈ ℝ<sup>6</sup>.
To simplify the point-cloud, we use K-Means to extract 100,000 centroids.

<table style="table-layout: fixed; width: 100%;">
<thead>
  <tr>
    <th colspan="2">For each video</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Top left: the rendered point cloud.<br></td>
    <td>Top right: The focal-plane trajectory for the predicted camera locations.</td>
  </tr>
  <tr>
    <td>Bottom left: The input video.</td>
    <td>Bottom right: The focal-plane trajectory of the ground truth camera locations.</td>
  </tr>
  <tr>
  <td colspan="2">
<a href="https://drive.google.com/file/d/13lFpLyqgsXcpgJn5iJtW8l7a-1TOEbiZ/view" title="RidgeSfm - mobile phone video frameskip k=3"><img src="cubot.jpg" alt="RidgeSfm - mobile phone video frameskip k=3" /></a>
</td>
  </tr>
</tbody>

<thead>
  <tr>
    <th colspan="2">Using MeshLab to display the point cloud:</th>
  </tr>
</thead>

<tr>
<td><img src="scene0_0.png" width="320" alt="ScanNet reconstruction" /></td>
<td><img src="scene0_1.png" width="320" alt="ScanNet reconstruction" /></td>
</tr>
<tr>
<td><img src="scene0_2.png" width="320" alt="ScanNet reconstruction" /></td>
<td><img src="scene0_3.png" width="320" alt="ScanNet reconstruction" /></td>
</tr>
</table>
