# Homebrew Formula for Alecto
# Place this in your homebrew-tap repository at Formula/alecto.rb
#
# Users install with:
#   brew tap rendro/tap
#   brew install alecto

class Alecto < Formula
  desc "Semantic memory for AI agents - local-first, MCP-native"
  homepage "https://github.com/rendro/alecto"
  version "0.1.0"

  on_macos do
    on_intel do
      url "https://github.com/rendro/alecto/releases/download/v#{version}/alecto-x86_64-apple-darwin.tar.gz"
      sha256 "REPLACE_WITH_ACTUAL_SHA256"
    end
    on_arm do
      url "https://github.com/rendro/alecto/releases/download/v#{version}/alecto-aarch64-apple-darwin.tar.gz"
      sha256 "REPLACE_WITH_ACTUAL_SHA256"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/rendro/alecto/releases/download/v#{version}/alecto-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "REPLACE_WITH_ACTUAL_SHA256"
    end
  end

  def install
    bin.install "alecto"
  end

  def caveats
    <<~EOS
      To use Alecto with Claude Desktop or Cursor:

      1. Initialize your project:
           cd your-project && alecto init

      2. Add to your MCP config:
           alecto config

      For Claude Desktop, add to:
        ~/Library/Application Support/Claude/claude_desktop_config.json

      For Cursor, add to your MCP settings.
    EOS
  end

  test do
    assert_match "Semantic memory for AI agents", shell_output("#{bin}/alecto --help")
  end
end
