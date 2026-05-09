import logoSrc from "@/assets/logo-liderancas.png";

export function BrandLogo({ size = 84, className = "" }: { size?: number; className?: string }) {
  return (
    <img
      src={logoSrc}
      alt="Lideranças Empáticas — FECAP"
      width={size}
      height={size}
      className={className}
      style={{ width: size, height: size, objectFit: "contain", display: "block" }}
    />
  );
}

export function FecapMark({ color = "#1F6B3F", height = 22 }: { color?: string; height?: number }) {
  return (
    <svg height={height} viewBox="0 0 240 60" style={{ display: "block" }}>
      <text
        x="0" y="48"
        fontFamily="Geist, system-ui, sans-serif"
        fontWeight="900" fontSize="56"
        fill={color}
        fontStyle="italic" letterSpacing="-2"
      >FECAP</text>
    </svg>
  );
}
