# Contributing to LLM Gateway

We love contributions! This document outlines how to contribute to the LLM Gateway project.

## 🚀 Getting Started

### Prerequisites

- Go 1.21 or higher
- Git
- A code editor (VS Code, GoLand, etc.)

### Setting Up Development Environment

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/yourusername/llm-gateway.git
cd llm-gateway

# Add upstream remote
git remote add upstream https://github.com/originaluser/llm-gateway.git

# Install dependencies
go mod download

# Run tests to make sure everything works
go test ./...

# Start the application
go run main.go
```

## 🛠️ Development Guidelines

### Code Style

- Follow standard Go formatting (`go fmt`)
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions small and focused

### Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for good test coverage

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run tests with race detection
go test -race ./...
```

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add support for new AI provider
fix: resolve rate limiting bug
docs: update README with new examples
test: add tests for authentication
```

## 📝 Pull Request Process

1. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, focused commits

3. **Test thoroughly**:
   ```bash
   go test ./...
   go build
   ```

4. **Update documentation** if needed

5. **Submit the PR** with:
   - Clear description of changes
   - Reference any related issues
   - Include screenshots for UI changes

## 🐛 Bug Reports

When reporting bugs, please include:

- Go version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

## 💡 Feature Requests

For new features:

- Check if it already exists in issues
- Describe the use case clearly
- Explain why it would be valuable
- Consider implementation approach

## 🔧 Development Areas

We welcome contributions in these areas:

### Core Features
- New AI provider adapters
- Performance optimizations
- Security enhancements
- Error handling improvements

### UI/UX
- Interface improvements
- Mobile responsiveness
- Accessibility features
- New dashboard features

### Documentation
- Code examples
- API documentation
- User guides
- Video tutorials

### Testing
- Unit tests
- Integration tests
- Performance benchmarks
- Security testing

## 🏗️ Architecture

### Project Structure

```
llm-gateway/
├── internal/
│   ├── adapter/       # AI provider adapters
│   ├── auth/         # Authentication logic
│   ├── config/       # Configuration management
│   ├── database/     # Database operations
│   ├── errors/       # Error handling
│   ├── router/       # Request routing
│   └── server/       # HTTP server
├── cmd/              # Command line tools
├── docs/             # Documentation
├── scripts/          # Build/deploy scripts
├── tests/            # Integration tests
├── ui.html           # Web interface
├── main.go           # Application entry point
└── models.yaml       # Model configuration
```

### Key Components

1. **Adapters** (`internal/adapter/`): Convert between OpenAI format and provider-specific formats
2. **Authentication** (`internal/auth/`): Handle OIDC/SSO integration
3. **Server** (`internal/server/`): HTTP server with routing and middleware
4. **Database** (`internal/database/`): Metrics storage and user management

## 🧪 Testing Strategy

### Unit Tests
- Test individual functions and methods
- Mock external dependencies
- Focus on edge cases

### Integration Tests
- Test API endpoints end-to-end
- Verify provider integrations
- Test authentication flows

### Performance Tests
- Benchmark critical paths
- Load testing for concurrent requests
- Memory usage profiling

## 📋 Code Review Checklist

Before submitting a PR, ensure:

- [ ] Code follows Go best practices
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes (or clearly documented)
- [ ] Error handling is appropriate
- [ ] Logging is meaningful
- [ ] Security considerations addressed

## 🎯 Release Process

1. **Version Bump**: Update version in relevant files
2. **Changelog**: Update CHANGELOG.md with new features/fixes
3. **Testing**: Full test suite passes
4. **Documentation**: Ensure docs are current
5. **Tag**: Create git tag with version
6. **Release**: Create GitHub release with notes

## 🆘 Getting Help

- **Discord**: [Join our community](https://discord.gg/llm-gateway)
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and ideas
- **Email**: maintainers@llm-gateway.dev

## 🙏 Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes
- GitHub repository contributors page

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to LLM Gateway! 🚀