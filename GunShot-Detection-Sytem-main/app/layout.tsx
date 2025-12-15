import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Gunshot Detection System',
  description: 'AI-Powered Security Monitoring System',
  generator: 'Gunshot Detection System',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body suppressHydrationWarning>{children}</body>
    </html>
  )
}
