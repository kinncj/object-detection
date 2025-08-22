---
name: Pull Request
about: Contribute changes to the object detection project
title: "[TYPE] Brief description of changes"
labels: ['needs review']
assignees: ['kinncj']
---

## 📋 Description

**What does this PR do?**
<!-- Briefly describe what this PR accomplishes -->

**Why is this change needed?**
<!-- Explain the motivation or issue this PR addresses -->

**What type of change is this?**
<!-- Check all that apply -->
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🧪 Test improvements
- [ ] 🔧 Code refactoring
- [ ] ⚡ Performance improvements
- [ ] 🎨 Code style changes
- [ ] 🔒 Security improvements

## 🧪 Testing

**How has this been tested?**
<!-- Describe the tests you ran and their results -->
- [ ] Existing tests pass (`pytest tests/`)
- [ ] New tests added for new functionality
- [ ] Manual testing performed
- [ ] Performance testing (if applicable)

**Test Coverage:**
```bash
# Run these commands to test your changes
pytest tests/ -v
pytest tests/ --cov=. --cov-report=term-missing
```

**Core Tests Status:**
<!-- These tests MUST pass for the PR to be merged -->
- [ ] `test_config.py` - Configuration management
- [ ] `test_drawer.py` - Drawing and visualization
- [ ] `test_factory.py` - Model factory patterns (critical)
- [ ] `test_frame_processor.py` - Frame processing logic

## 📱 Environment

**Development Environment:**
- OS: <!-- e.g., macOS, Ubuntu 22.04, Windows 11 -->
- Python Version: <!-- e.g., 3.8, 3.9, 3.10 -->
- Environment Type: <!-- conda, venv, virtualenv -->
- Key Dependencies: <!-- Any specific versions used -->

**Setup Command Used:**
```bash
# Command used to set up the environment
conda env create -f environment.yml  # or other method
```

## 🔍 Code Review Checklist

**Self-Review Completed:**
- [ ] Code follows project style guidelines (black, flake8)
- [ ] Self-reviewed the code changes
- [ ] Added comments for complex logic
- [ ] Updated documentation as needed
- [ ] Removed debug code and print statements
- [ ] No hardcoded values or sensitive information

**AI Tools Used:**
- [ ] GitHub Copilot assisted with code generation
- [ ] AI-generated code has been reviewed and tested
- [ ] Comments added to explain AI-suggested complex logic

## 📊 Performance Impact

**Performance Considerations:**
<!-- If applicable, describe performance implications -->
- [ ] No significant performance impact
- [ ] Performance improvements included
- [ ] Performance tested with benchmarks
- [ ] Memory usage considered

**Benchmarks (if applicable):**
```
Before: [describe performance metrics]
After:  [describe performance metrics]
```

## 🔗 Related Issues

**Fixes/Closes:**
<!-- Link to issues this PR addresses -->
- Fixes #<!-- issue number -->
- Closes #<!-- issue number -->

**Related PRs:**
<!-- Link to related PRs -->
- Related to #<!-- PR number -->

## 📸 Screenshots/Recordings

**Visual Changes:**
<!-- If your changes affect the UI or output, include screenshots or recordings -->

**Before:**
<!-- Screenshot or description of current behavior -->

**After:**
<!-- Screenshot or description of new behavior -->

## 🚀 Deployment Notes

**Breaking Changes:**
<!-- List any breaking changes and migration steps -->
- [ ] No breaking changes
- [ ] Breaking changes documented below

**Migration Steps:**
```bash
# If there are breaking changes, provide migration steps
# Example: Update configuration files, run migration scripts, etc.
```

**Dependencies:**
- [ ] No new dependencies added
- [ ] New dependencies added (listed in requirements.txt/environment.yml)
- [ ] Dependencies removed (cleanup performed)

## 📚 Documentation

**Documentation Updates:**
- [ ] README.md updated
- [ ] SETUP_GUIDE.md updated
- [ ] Code comments added/updated
- [ ] API documentation updated
- [ ] No documentation changes needed

## 🤖 AI Review Summary

<!-- This section will be automatically filled by the AI review workflow -->
**Automated Review Status:**
- [ ] Code quality checks passed
- [ ] Security scan completed
- [ ] Test coverage maintained
- [ ] Performance impact assessed

---

## 📝 Additional Notes

<!-- Any additional information for reviewers -->

**For Reviewers:**
<!-- Specific areas you'd like reviewers to focus on -->

**Questions for Maintainers:**
<!-- Any questions or concerns about the implementation -->

---

### 🤖 GitHub Copilot Integration

This PR will be automatically reviewed by GitHub Copilot and other AI tools to check for:
- Code quality and best practices
- Potential security vulnerabilities
- Performance considerations
- Test coverage gaps
- Documentation completeness

**Human reviewers should focus on:**
- Business logic correctness
- Architecture decisions
- User experience impact
- Integration considerations
