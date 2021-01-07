package com.gwjjj.weiborumor.Domain;


import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Setter
@Getter
public class CommentReturn {
    private List<Comment> comments;

    private String prob;
}
