import type { Metadata } from "next";
import { Be_Vietnam_Pro } from "next/font/google";
import "./globals.css";

const beVietnamPro = Be_Vietnam_Pro({
  variable: "--font-be-vietnam-pro",
  subsets: ["latin", "vietnamese"],
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "Unimate — Trợ lý sinh viên ĐH Giao thông Vận tải",
  description: "Unimate - Trợ lý AI tra cứu quy chế đào tạo tín chỉ, học bổng, điểm rèn luyện và học vụ Sổ tay K67 UTC.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="vi" className={beVietnamPro.variable}>
      <body>{children}</body>
    </html>
  );
}
