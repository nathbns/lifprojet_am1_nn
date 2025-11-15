"use client";

import { useEffect, useState, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { useGLTF, OrbitControls, Bounds } from '@react-three/drei';
import * as THREE from 'three';

// Composant pour charger et afficher le modèle GLB
function RookModel({ color }: { color: string }) {
  const { scene } = useGLTF('/004_chess_rook.glb');

  // Appliquer la couleur au modèle
  useEffect(() => {
    if (scene) {
      scene.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.material = new THREE.MeshStandardMaterial({
            color: color,
            metalness: 0.8,
            roughness: 0.2,
            side: THREE.DoubleSide,
          });
        }
      });
    }
  }, [scene, color]);

  return (
    <group rotation={[0.2, 0, 0]} scale={0.02} position={[0, -0.3, 0]}>
      <primitive object={scene} />
    </group>
  );
}

// Précharger le modèle
useGLTF.preload('/004_chess_rook.glb');

export default function ChessPiece3D() {
  const [pieceColor, setPieceColor] = useState('#000000');

  useEffect(() => {
    // Fonction pour mettre à jour la couleur selon le thème
    const updateColor = () => {
      const isDarkMode = document.documentElement.classList.contains('dark');
      // Noir en mode clair, blanc en mode sombre
      setPieceColor(isDarkMode ? '#ffffff' : '#000000');
    };

    // Mettre à jour au chargement
    updateColor();

    // Observer les changements de thème
    const observer = new MutationObserver(updateColor);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    });

    return () => observer.disconnect();
  }, []);

  return (
    <div className="w-48 h-48 sm:w-24 sm:h-24 md:w-36 md:h-36 lg:w-48 lg:h-48 xl:w-64 xl:h-64">
      <Canvas 
        camera={{ 
          position: [0, -0.5, 4], 
          fov: 50,
          up: [0, 1, 0]
        }} 
        shadows
      >
        <ambientLight intensity={1.5} />
        <directionalLight 
          position={[5, 5, 5]} 
          intensity={2} 
          castShadow 
        />
        <directionalLight position={[-5, 3, -3]} intensity={1} />
        <spotLight position={[0, 8, 0]} intensity={3} angle={0.6} penumbra={0.5} />
        <pointLight position={[0, 5, 3]} intensity={1.5} />
        <hemisphereLight intensity={0.8} />
        
        <Suspense fallback={null}>
          <Bounds fit clip observe margin={1.4}>
            <RookModel color={pieceColor} />
          </Bounds>
        </Suspense>
        
        <OrbitControls 
          enableZoom={false} 
          enablePan={false}
          autoRotate={true}
          autoRotateSpeed={2}
          makeDefault
          target={[0, 0, 0]}
        />
      </Canvas>
    </div>
  );
}

