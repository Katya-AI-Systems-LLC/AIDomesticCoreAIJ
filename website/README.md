# AIPlatform Quantum Infrastructure Zero SDK - Multilingual Website

## Overview

This is the multilingual website for the AIPlatform Quantum Infrastructure Zero SDK. The website supports four languages:
- English (en)
- Russian (ru)
- Chinese (zh)
- Arabic (ar)

## Features

### Multilingual Support
- Full internationalization with language selector
- Right-to-left (RTL) support for Arabic
- Responsive design that works in all languages
- Language preference saving in localStorage

### Technical Features
- Modern responsive design
- Smooth animations and transitions
- Quantum particle background effects
- Interactive elements with hover effects
- Performance monitoring
- Cross-browser compatibility

## Languages

### English
The default language with full technical terminology.

### Russian (Русский)
Complete translation with proper Cyrillic support and technical vocabulary.

### Chinese (中文)
Full translation with proper character encoding and technical terms.

### Arabic (العربية)
Complete translation with RTL text support and proper Arabic formatting.

## File Structure

```
website/
├── index.html          # Main HTML file with multilingual support
├── styles.css          # CSS with RTL and responsive design
├── script.js           # JavaScript for interactivity
├── i18n.js             # Internationalization library
└── README.md           # This file
```

## Implementation Details

### Language Selector
Located in the top-right corner, allows users to switch between languages instantly.

### Data Attributes
All translatable content uses `data-i18n` attributes for easy localization:
```html
<h1 data-i18n="hero.title">Quantum-AI Infrastructure<br><span class="highlight">Zero SDK</span></h1>
```

### Translation System
The `i18n.js` file contains all translations in a structured JSON format:
```javascript
const translations = {
    en: {
        hero: {
            title: "Quantum-AI Infrastructure<br><span class=\"highlight\">Zero SDK</span>"
        }
    },
    ru: {
        hero: {
            title: "Квантово-ИИ Инфраструктура<br><span class=\"highlight\">Zero SDK</span>"
        }
    }
    // ... other languages
};
```

### RTL Support
Arabic language automatically enables RTL layout:
```javascript
document.documentElement.setAttribute('dir', lang === 'ar' ? 'rtl' : 'ltr');
```

## Usage

1. Open `index.html` in a web browser
2. Select your preferred language from the top-right dropdown
3. The entire website will instantly translate to your selected language
4. Your language preference is saved automatically

## Technical Requirements

- Modern web browser with JavaScript enabled
- CSS3 support for animations and transitions
- localStorage support for saving language preferences

## Customization

### Adding New Languages
1. Add the language code to the language selector in `index.html`
2. Add translations to the `translations` object in `i18n.js`
3. Ensure proper RTL support if needed

### Adding New Content
1. Add `data-i18n` attributes to new HTML elements
2. Add corresponding translations to all languages in `i18n.js`

## Browser Support

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Performance

- Lightweight implementation (~15KB total)
- No external dependencies
- Efficient translation lookup
- Smooth animations with hardware acceleration

## Accessibility

- Proper semantic HTML structure
- Keyboard navigation support
- Screen reader friendly
- High contrast mode compatible

## Security

- No external resources loaded
- No tracking or analytics
- Client-side only implementation
- No user data collection

## License

This website is part of the AIPlatform Quantum Infrastructure Zero SDK project and is distributed under the same license as the main project.