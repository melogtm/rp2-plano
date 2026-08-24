;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "12pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("sbc-template" "") ("graphicx" "") ("url" "") ("float" "") ("inputenc" "utf8") ("babel" "brazil")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "url")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art12"
    "sbc-template"
    "graphicx"
    "url"
    "float"
    "inputenc"
    "babel")
   (TeX-add-symbols
    '("inst" 1))
   (LaTeX-add-labels
    "tab:cronograma")
   (LaTeX-add-bibliographies
    "referencias"))
 :latex)

