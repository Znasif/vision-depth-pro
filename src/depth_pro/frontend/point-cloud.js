import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export class PointCloudViewer {
    constructor(container) {
        this.container = container;
        this.init();
    }

    init() {
        // Setup scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf0f0f0);

        // Setup camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.z = 5;

        // Setup renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.container.appendChild(this.renderer.domElement);

        // Setup controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        // Add coordinate frame
        const axesHelper = new THREE.AxesHelper(2);
        this.scene.add(axesHelper);

        // Start animation loop
        this.animate();
    }

    updatePointCloud(points, colors) {
        // Remove existing point cloud if any
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
        }

        // Create geometry
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        // Create material
        const material = new THREE.PointsMaterial({
            size: 0.02,
            vertexColors: true,
            sizeAttenuation: true
        });

        // Create point cloud
        this.pointCloud = new THREE.Points(geometry, material);
        this.scene.add(this.pointCloud);

        // Adjust camera
        const box = new THREE.Box3().setFromObject(this.pointCloud);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const fov = this.camera.fov * (Math.PI / 180);
        const cameraDistance = Math.abs(maxDim / Math.sin(fov / 2));

        this.camera.position.copy(center);
        this.camera.position.z += cameraDistance;
        this.camera.lookAt(center);

        this.controls.target.copy(center);
        this.controls.update();
    }

    animate = () => {
        requestAnimationFrame(this.animate);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    dispose() {
        this.renderer.dispose();
        this.container.removeChild(this.renderer.domElement);
    }
}